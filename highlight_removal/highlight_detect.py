from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .face_regions import RegionMasks
from .utils import blur_mask, mask_to_uint8, morph


@dataclass
class HighlightOutput:
    hard_mask: np.ndarray
    soft_mask: np.ndarray
    debug: Dict[str, np.ndarray]
    warnings: List[str]


def _p(params: Dict[str, Any], names, default):
    """读取参数，兼容旧版 default.yaml 中的别名。"""
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in params:
            return params[name]
    return default


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (mask_to_uint8(mask) > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == i] = 255
    return out


def _masked_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    values_f = values.astype(np.float32)
    valid_f = valid.astype(np.float32)
    num = cv2.GaussianBlur(values_f * valid_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    den = cv2.GaussianBlur(valid_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return num / np.maximum(den, 1e-3)


def _percentile(values: np.ndarray, q: float, fallback: float) -> float:
    if values is None or values.size == 0:
        return float(fallback)
    return float(np.percentile(values.astype(np.float32), float(q)))


def _grow_from_core(core: np.ndarray, halo: np.ndarray, max_iter: int) -> np.ndarray:
    """从强高光核心向同一局部候选晕区扩展，不允许整脸漫延。"""
    grown = core.astype(bool).copy()
    halo_bool = halo.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(0, int(max_iter))):
        nxt = cv2.dilate(grown.astype(np.uint8), kernel, iterations=1).astype(bool) & halo_bool
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def _cap_by_score(mask: np.ndarray, score: np.ndarray, region: np.ndarray, max_pixels: int) -> np.ndarray:
    """按高光强度保留区域内最可信像素，防止 mask 变成整片皮肤。"""
    out = mask.astype(bool) & region.astype(bool)
    count = int(out.sum())
    if max_pixels <= 0 or count <= max_pixels:
        return out
    vals = score[out]
    if vals.size <= max_pixels:
        return out
    kth = np.partition(vals, -int(max_pixels))[-int(max_pixels)]
    return out & (score >= kth)


def _region_fraction(key: str, params: Dict[str, Any]) -> float:
    if key == "forehead":
        return float(_p(params, "forehead_region_max_fraction", 0.30))
    if key == "nose_bridge":
        return float(_p(params, "nose_bridge_region_max_fraction", 0.72))
    if key == "nose_tip":
        return float(_p(params, "nose_tip_region_max_fraction", 0.55))
    if key in ("left_cheek", "right_cheek"):
        return float(_p(params, "cheek_region_max_fraction", 0.20))
    if key == "chin":
        return float(_p(params, "chin_region_max_fraction", 0.24))
    if key == "philtrum":
        return float(_p(params, "philtrum_region_max_fraction", 0.32))
    if key in ("left_brow_ridge", "right_brow_ridge"):
        return float(_p(params, "brow_region_max_fraction", 0.16))
    if key in ("highlight_candidate", "treatable_skin"):
        return float(_p(params, "skin_region_max_fraction", 0.35))
    return 0.20


def _adaptive_region_mask(
    key: str,
    region_mask: np.ndarray,
    base: np.ndarray,
    rgb_max: np.ndarray,
    rgb_range: np.ndarray,
    gray: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    l: np.ndarray,
    chroma_delta: np.ndarray,
    local_diff: np.ndarray,
    large_diff: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """区域自适应高光检测。

    证件照中额头/鼻梁/鼻尖油光常是暖黄色，不一定低饱和或纯白。
    因此这里不再要求“白色 + 低饱和”，而是对每个关键点区域分别建立亮度基线：
    - 核心：区域内亮度分位数很高或局部亮度突增；
    - 晕区：比同一区域正常皮肤更亮，且不越过皮肤/保护区；
    - 最终：从核心向晕区生长，再按得分和面积上限截断。
    """
    region = (mask_to_uint8(region_mask) > 0) & base
    if int(region.sum()) < 20:
        z = np.zeros_like(base, dtype=bool)
        return z, z, np.zeros_like(l, dtype=np.float32), {}

    lv = l[region]
    vv = v[region]
    sv = s[region]
    gv = gray[region]
    cdv = chroma_delta[region]
    ldv = local_diff[region]

    l50 = _percentile(lv, 50, 185)
    l75 = _percentile(lv, 75, l50 + 7)
    l90 = _percentile(lv, float(_p(params, "adaptive_core_percentile", 90)), l50 + 14)
    l95 = _percentile(lv, 95, l90 + 5)
    v50 = _percentile(vv, 50, 205)
    v75 = _percentile(vv, 75, v50 + 7)
    v90 = _percentile(vv, 90, v50 + 14)
    s50 = _percentile(sv, 50, 95)
    s75 = _percentile(sv, 75, s50 + 10)
    g50 = _percentile(gv, 50, 175)
    g75 = _percentile(gv, 75, g50 + 7)
    g90 = _percentile(gv, 90, g50 + 14)
    cd75 = _percentile(cdv, 75, 24)
    ld75 = _percentile(ldv, 75, 2)
    ld90 = _percentile(ldv, 90, 5)

    abs_l_thr = float(_p(params, ["lab_l_threshold", "lab_l_thr"], 188))
    abs_v_thr = float(_p(params, ["hsv_v_threshold", "hsv_v_thr"], 190))
    rgb_thr = float(_p(params, ["rgb_brightness_threshold", "rgb_thr"], 205))
    oil_s_upper = float(_p(params, "oil_shine_s_upper", max(150.0, float(_p(params, "hsv_s_upper", 150)))))
    hsv_s_upper = float(_p(params, "hsv_s_upper", 150))
    delta = float(_p(params, "adaptive_region_delta", 4.8))
    core_delta = float(_p(params, "adaptive_core_delta", 8.0))
    rel_sat_drop = float(_p(params, "relative_saturation_drop", -8.0))
    halo_pct = float(_p(params, "adaptive_halo_percentile", 78))
    core_pct = float(_p(params, "adaptive_core_percentile", 90))

    l_halo_thr = max(_percentile(lv, halo_pct, l75), l50 + delta)
    l_core_thr = max(_percentile(lv, core_pct, l90), l50 + core_delta)
    v_halo_thr = max(_percentile(vv, halo_pct, v75), v50 + delta)
    g_halo_thr = max(_percentile(gv, halo_pct, g75), g50 + delta)

    # 对鼻梁/鼻尖更敏感；对脸颊/下巴稍保守。
    if key in ("nose_bridge", "nose_tip"):
        l_halo_thr -= 2.0
        v_halo_thr -= 2.0
        g_halo_thr -= 2.0
        l_core_thr -= 2.0
    elif key in ("left_brow_ridge", "right_brow_ridge"):
        l_halo_thr -= 2.0
        v_halo_thr -= 2.0
        g_halo_thr -= 2.0
        l_core_thr -= 2.5
    elif key in ("left_cheek", "right_cheek"):
        l_halo_thr += 1.5
        l_core_thr += 1.0
    elif key == "chin":
        l_halo_thr += 1.0

    # 归一化高光得分。暖色油光的 S 可能较高，所以低饱和只作为加分，不作为硬门槛。
    l_norm = (l - l50) / max(l95 - l50, 8.0)
    v_norm = (v - v50) / max(v90 - v50, 8.0)
    g_norm = (gray - g50) / max(g90 - g50, 8.0)
    local_norm = np.maximum(local_diff, large_diff * 0.65) / max(max(ld90, ld75 + 1.0), 3.5)
    white_score = ((rgb_range <= 62).astype(np.float32) * 0.55 + (s <= max(hsv_s_upper, s50 + 30)).astype(np.float32) * 0.25)
    abs_score = ((l >= abs_l_thr).astype(np.float32) + (v >= abs_v_thr).astype(np.float32) + (rgb_max >= rgb_thr).astype(np.float32)) / 3.0
    chroma_score = (chroma_delta <= max(float(_p(params, "local_skin_chroma_delta_upper", 42)), cd75 + 14)).astype(np.float32)

    score = (
        0.40 * np.clip(l_norm, 0, 2)
        + 0.22 * np.clip(v_norm, 0, 2)
        + 0.13 * np.clip(g_norm, 0, 2)
        + 0.16 * np.clip(local_norm, 0, 2)
        + 0.12 * white_score
        + 0.18 * abs_score
        + 0.08 * chroma_score
    ).astype(np.float32)

    warm_allowed = s <= max(oil_s_upper, s75 + 42.0)
    not_too_colored = chroma_delta <= max(float(_p(params, "local_skin_chroma_delta_upper", 42)), cd75 + 18.0)
    relative_low_sat = s <= (s50 - rel_sat_drop)  # rel_sat_drop 可为负值，表示允许比中位数略高。

    absolute_core = (
        ((l >= abs_l_thr) & (v >= abs_v_thr - 4) & warm_allowed)
        | ((rgb_max >= rgb_thr) & (l >= abs_l_thr - 8) & warm_allowed)
        | ((rgb_max >= float(_p(params, "saturation_pixel_threshold", 245)) - 3) & (rgb_range <= 92))
    )
    percentile_core = region & (l >= l_core_thr) & (v >= v_halo_thr - 2) & warm_allowed
    local_core = region & (local_diff >= max(float(_p(params, "local_brightness_threshold", 5)), ld75 + 0.5)) & (l >= l_halo_thr - 2) & warm_allowed
    core = region & not_too_colored & (absolute_core | percentile_core | local_core | ((score >= 0.74) & relative_low_sat))

    # 如果高光是一片平滑油光，局部差异不强，core 可能为空；用区域内得分最高的极小部分作为种子。
    if int(core.sum()) == 0 and (l95 >= abs_l_thr - 5 or v90 >= abs_v_thr - 4 or _percentile(rgb_max[region], 95, 0) >= rgb_thr - 4):
        seed_pct = float(_p(params, "fallback_seed_percentile", 93))
        seed_thr = _percentile(score[region], seed_pct, 0.85)
        core = region & not_too_colored & (score >= seed_thr) & (l >= l_halo_thr - 1) & warm_allowed

    halo = region & not_too_colored & warm_allowed & (
        (l >= l_halo_thr)
        | (v >= v_halo_thr)
        | (gray >= g_halo_thr)
        | ((score >= 0.46) & (l >= l50 + delta * 0.55))
        | ((large_diff >= max(2.0, float(_p(params, "local_brightness_threshold", 5)) * 0.50)) & (l >= l50 + delta * 0.5))
    )

    grown = _grow_from_core(core, halo, int(_p(params, "adaptive_grow_iterations", 12)))
    max_fraction = np.clip(_region_fraction(key, params), 0.03, 0.95)
    max_pixels = max(6, int(region.sum() * max_fraction))
    grown = _cap_by_score(grown | core, score, region, max_pixels=max_pixels)
    core = _cap_by_score(core, score, region, max_pixels=max(3, int(max_pixels * 0.55)))

    stats = {
        "l50": l50,
        "l75": l75,
        "l90": l90,
        "l95": l95,
        "v50": v50,
        "v90": v90,
        "s50": s50,
        "s75": s75,
        "l_halo_thr": float(l_halo_thr),
        "l_core_thr": float(l_core_thr),
    }
    return core, grown, score, stats


def detect_highlight(image_bgr: np.ndarray, regions: RegionMasks, params: Dict[str, Any]) -> HighlightOutput:
    warnings: List[str] = []
    h, w = image_bgr.shape[:2]

    skin = mask_to_uint8(regions.masks.get("skin", np.zeros((h, w), np.uint8))) > 0
    protect = mask_to_uint8(regions.masks.get("protect", np.zeros((h, w), np.uint8))) > 0
    candidate_region = mask_to_uint8(regions.masks.get("highlight_candidate", np.zeros((h, w), np.uint8))) > 0
    base = skin & candidate_region & (~protect)

    if int(base.sum()) == 0:
        z = np.zeros((h, w), np.uint8)
        return HighlightOutput(z, z, {}, ["高光候选区域为空，未执行高光检测"])

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rgb_max = rgb.max(axis=2).astype(np.float32)
    rgb_min = rgb.min(axis=2).astype(np.float32)
    rgb_range = rgb_max - rgb_min
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    rgb_thr = float(_p(params, ["rgb_brightness_threshold", "rgb_thr"], 205))
    hsv_v_thr = float(_p(params, ["hsv_v_threshold", "hsv_v_thr"], 190))
    hsv_s_upper = float(_p(params, "hsv_s_upper", 150))
    oil_s_upper = float(_p(params, "oil_shine_s_upper", max(165.0, hsv_s_upper)))
    lab_l_thr = float(_p(params, ["lab_l_threshold", "lab_l_thr"], 188))
    sat_thr = float(_p(params, ["saturation_pixel_threshold", "saturated_pixel_threshold"], 245))
    local_thr = float(_p(params, ["local_brightness_threshold", "local_brightness_delta"], 5))
    local_ratio_thr = float(_p(params, "local_contrast_threshold", 0.025))

    # 局部亮度基线只由有效人脸皮肤候选区计算，避免背景/衣服影响阈值。
    local_l = _masked_gaussian(l, base, float(_p(params, "local_sigma", 6.0)))
    large_l = _masked_gaussian(l, base, float(_p(params, "large_local_sigma", 28.0)))
    local_diff = l - local_l
    large_diff = l - large_l
    local_ratio = local_diff / np.maximum(local_l, 1.0)

    # 肤色邻域色差：仅作为极端排除条件，不再把暖色油光误删。
    win = int(_p(params, "local_window_radius", 31))
    win = max(7, win if win % 2 == 1 else win + 1)
    base_f = base.astype(np.float32)
    w_sum = cv2.GaussianBlur(base_f, (win, win), 0) + 1e-3
    a_local = cv2.GaussianBlur(a * base_f, (win, win), 0) / w_sum
    b_local = cv2.GaussianBlur(b * base_f, (win, win), 0) / w_sum
    chroma_delta = np.sqrt((a - a_local) ** 2 + (b - b_local) ** 2)
    chroma_ok = chroma_delta <= float(_p(params, "local_skin_chroma_delta_upper", 42))

    # 传统强证据：纯白点、低饱和高亮、Lab 高亮、局部突增。
    rule_rgb = (rgb_max >= rgb_thr) | ((rgb_max >= sat_thr) & (rgb_range <= 96))
    rule_hsv_white = (v >= hsv_v_thr) & (s <= hsv_s_upper)
    rule_hsv_oil = (v >= hsv_v_thr - 8) & (s <= oil_s_upper)
    rule_lab = l >= lab_l_thr
    rule_local = (local_diff >= local_thr) | (local_ratio >= local_ratio_thr) | (large_diff >= max(2.0, local_thr * 0.45))
    score_count = rule_rgb.astype(np.uint8) + rule_hsv_white.astype(np.uint8) + rule_hsv_oil.astype(np.uint8) + rule_lab.astype(np.uint8) + rule_local.astype(np.uint8)
    classic = base & chroma_ok & (
        ((score_count >= 2) & (rule_lab | rule_hsv_oil | rule_local))
        | ((score_count >= 1) & rule_local & (rule_rgb | rule_lab | rule_hsv_oil))
    )

    adaptive_core = np.zeros((h, w), dtype=bool)
    adaptive_grown = np.zeros((h, w), dtype=bool)
    global_score = np.zeros((h, w), dtype=np.float32)
    debug: Dict[str, np.ndarray] = {}
    region_stats: Dict[str, Dict[str, float]] = {}

    region_keys = (
        ["highlight_candidate"]
        if "treatable_skin" in regions.masks
        else [
            "forehead", "nose_bridge", "nose_tip", "philtrum",
            "left_brow_ridge", "right_brow_ridge",
            "left_cheek", "right_cheek", "chin",
        ]
    )
    for key in region_keys:
        if key not in regions.masks:
            continue
        core_i, grown_i, score_i, stats_i = _adaptive_region_mask(
            key, regions.masks[key], base, rgb_max, rgb_range, gray, v, s, l, chroma_delta,
            local_diff, large_diff, params
        )
        adaptive_core |= core_i
        adaptive_grown |= grown_i
        global_score = np.maximum(global_score, score_i)
        debug[f"adaptive_core_{key}"] = core_i.astype(np.uint8) * 255
        debug[f"adaptive_grown_{key}"] = grown_i.astype(np.uint8) * 255
        region_stats[key] = stats_i

    mask_bool = base & (classic | adaptive_core | adaptive_grown)
    hard = mask_bool.astype(np.uint8) * 255

    # 形态学只做闭合和轻微扩张，不做默认腐蚀，防止额头/鼻尖高光核心被吞掉。
    close_r = int(_p(params, "morph_close_radius", 3))
    dilate_r = int(_p(params, ["mask_dilate_radius", "morph_dilate_radius"], 1))
    erode_r = int(_p(params, ["mask_erode_radius", "morph_erode_radius"], 0))
    blur_r = int(_p(params, "mask_blur_radius", 9))

    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, k, iterations=1)
    if erode_r > 0:
        hard = morph(hard, "erode", erode_r)
    if dilate_r > 0:
        hard = morph(hard, "dilate", dilate_r)

    hard = cv2.bitwise_and(hard, (base.astype(np.uint8) * 255))
    hard[protect] = 0
    hard = _remove_small_components(hard, int(_p(params, ["highlight_min_area", "min_area"], 8)))

    face_area = max(1, int((mask_to_uint8(regions.masks.get("face_mask", regions.masks.get("skin", hard))) > 0).sum()))
    max_area = int(face_area * float(_p(params, ["highlight_max_area_ratio", "max_area_ratio"], 0.125)))
    if max_area > 0 and int((hard > 0).sum()) > max_area:
        # 只截断，不立即报错；真正超过真实性阈值由 quality_metrics 统一警告。
        strength = global_score + 0.18 * np.clip((l - lab_l_thr) / 20.0, 0, 3) + 0.12 * np.clip((v - hsv_v_thr) / 25.0, 0, 3) + 0.10 * np.clip(local_diff / max(local_thr, 1.0), 0, 3)
        vals = strength[hard > 0]
        if vals.size > max_area:
            kth = np.partition(vals, -max_area)[-max_area]
            hard = ((hard > 0) & (strength >= kth) & base).astype(np.uint8) * 255
            hard[protect] = 0

    if int((hard > 0).sum()) == 0:
        warnings.append("未检测到满足条件的高光区域，可降低 HSV V、Lab L、局部亮度异常阈值或提高油光饱和度上限")

    if np.any((hard > 0) & protect):
        warnings.append("眼睛或嘴唇区域被误选中，已自动从高光 mask 中排除")
        hard[protect] = 0

    soft = blur_mask(hard, blur_r) if blur_r > 0 else hard.copy()
    soft_gain = float(_p(params, "soft_mask_gain", 1.65))
    soft = np.clip(soft.astype(np.float32) * soft_gain, 0, 255).astype(np.uint8)
    soft = cv2.bitwise_and(soft, (base.astype(np.uint8) * 255))
    soft[protect] = 0

    likelihood = np.clip(global_score / max(float(global_score.max()), 1e-3) * 255, 0, 255).astype(np.uint8)
    debug.update({
        "rule_rgb": (rule_rgb & base).astype(np.uint8) * 255,
        "rule_hsv_white": (rule_hsv_white & base).astype(np.uint8) * 255,
        "rule_hsv_oil": (rule_hsv_oil & base).astype(np.uint8) * 255,
        "rule_lab": (rule_lab & base).astype(np.uint8) * 255,
        "rule_local": (rule_local & base).astype(np.uint8) * 255,
        "adaptive_core": adaptive_core.astype(np.uint8) * 255,
        "adaptive_grown": adaptive_grown.astype(np.uint8) * 255,
        "candidate_base": base.astype(np.uint8) * 255,
        "highlight_likelihood": cv2.applyColorMap(likelihood, cv2.COLORMAP_JET),
        "local_diff": np.clip(local_diff + 128, 0, 255).astype(np.uint8),
    })

    return HighlightOutput(hard_mask=hard, soft_mask=soft, debug=debug, warnings=warnings)


def refine_upsampled_masks(
    image_bgr: np.ndarray,
    regions: RegionMasks,
    hard_seed: np.ndarray,
    soft_seed: np.ndarray,
    params: Dict[str, Any],
    process_scale: float,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """低分辨率 mask 上采样后，在局部 ROI 内做全分辨率高光重检测，避免大块灰斑。"""
    from .utils import bbox_from_mask, compute_roi_bbox, merge_bboxes

    warnings: List[str] = []
    h, w = image_bgr.shape[:2]
    hard_s = mask_to_uint8(hard_seed)
    soft_s = mask_to_uint8(soft_seed)
    hard_seed_pixels = int((hard_s > 0).sum())
    seed_pixels = hard_seed_pixels + int((soft_s > 48).sum())
    candidate_mask = regions.masks.get("highlight_candidate", regions.masks.get("skin"))

    pad_ratio = 0.18 + 0.12 * (1.0 - process_scale)
    roi_seed = compute_roi_bbox(hard_s, soft_s, image_bgr.shape, padding_ratio=pad_ratio)
    roi_cand = bbox_from_mask(candidate_mask, image_bgr.shape, padding_ratio=max(pad_ratio, 0.22))
    if roi_seed is None or hard_seed_pixels == 0:
        roi = roi_cand
        if roi is None:
            return hard_seed, soft_seed, warnings
        warnings.append("低分辨率高光 seed 偏弱，已改用全脸候选区做全分辨率重检测")
    elif process_scale <= 0.26:
        roi = merge_bboxes(roi_seed, roi_cand)
        if roi is None:
            return hard_seed, soft_seed, warnings
    else:
        roi = roi_seed

    x, y, rw, rh = roi
    crop = image_bgr[y : y + rh, x : x + rw]
    crop_regions = RegionMasks(
        {k: mask_to_uint8(v)[y : y + rh, x : x + rw] for k, v in regions.masks.items()},
        regions.geometry,
        regions.warnings,
    )
    refined = detect_highlight(crop, crop_regions, params)
    hard = np.zeros((h, w), np.uint8)
    soft = np.zeros((h, w), np.uint8)
    hard[y : y + rh, x : x + rw] = mask_to_uint8(refined.hard_mask)
    soft[y : y + rh, x : x + rw] = mask_to_uint8(refined.soft_mask)

    skin = mask_to_uint8(regions.masks.get("skin", np.zeros((h, w), np.uint8)))
    protect = mask_to_uint8(regions.masks.get("protect", np.zeros((h, w), np.uint8)))
    candidate = mask_to_uint8(regions.masks.get("highlight_candidate", skin))
    allowed = (candidate > 0) & (skin > 0) & (protect == 0)
    hard = np.where(allowed, hard, 0).astype(np.uint8)
    soft = np.where(allowed, soft, 0).astype(np.uint8)

    # 仅在 seed 足够可靠时，用它限制外扩；0.25 等弱 seed 不再硬裁剪，避免漏检。
    if seed_pixels >= 256 and process_scale >= 0.5:
        pad = max(2, int(round(4 * (1.0 - process_scale))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        seed = cv2.dilate(np.maximum(hard_s, soft_s), k, iterations=1)
        hard = cv2.bitwise_and(hard, seed)
        soft = cv2.bitwise_and(soft, seed)

    warnings.extend(refined.warnings)
    return hard, soft, warnings
