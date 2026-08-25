"""B 部分：多方面 A/B 实验（固定 landmarks，不改生产默认路径）。

所有变体使用 experiments/run_audit.py 存下的同一套 478 点关键点，
排除人脸检测噪声；评估 mask（treatable/protect/face_mask 与
「输入定义的油光候选区」）全部来自基线，保证跨变体可比。

变体一览（base = 常用模式 = 检测「灵敏」+ 修复「强力」+ compromise 分辨率）：
  keyregion        regions.key_region_detection_mode=true（分区自适应）
  det{灵敏|正常|弱} x rm{强力|正常|削弱} 九宫格（base 即 灵敏x强力）
  no_extreme       关掉混合模式的 extreme Telea 核心（extreme_core_extra_l=999）
  drm_minrgb       经典 DRM：I_spec ≈ minRGB - 邻域皮肤 minRGB 基线，仅 soft mask 内减除
  guided_filter    修复参考图中 bilateralFilter → ximgproc.guidedFilter
  log_shine_or     Y 通道 LoG 油光候选 OR 进现有 hard mask
  chroma_up        chroma_restore 0.38→0.55、brightness_suppress 0.94→0.80（减发灰）
  fullres          pipeline.process_scale=1.0（其余与 base 相同）

用法：.venv/bin/python experiments/run_experiments.py [变体名 ...]
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    NAMES,
    OUT_DIR,
    dump_json,
    imread,
    lap_var,
    load_face,
    masked_ssim,
    residual_shine,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cli_process import INTENSITY_PATH, SENSITIVITY_PATH, _merge_preset, load_cli_config  # noqa: E402
from highlight_removal import remove_highlight as rh_mod  # noqa: E402
from highlight_removal.pipeline import _detect_highlight_masks, _process_core, _removal_params  # noqa: E402
from highlight_removal.quality_metrics import compute_metrics  # noqa: E402
from highlight_removal.remove_highlight import RemovalOutput, remove_highlight  # noqa: E402
from highlight_removal.utils import mask_to_uint8, normalize_mask  # noqa: E402

VAR_DIR = OUT_DIR / "variants"


# ---------------------------------------------------------------- 配置变体

def cfg_base() -> dict:
    return load_cli_config("常用模式")


def cfg_grid(det: str, rm: str) -> dict:
    cfg = cfg_base()
    base = load_cli_config("高保真模式")  # 正常/正常的干净底
    cfg["highlight_detection"] = _merge_preset(deepcopy(base["highlight_detection"]), SENSITIVITY_PATH, det)
    cfg["highlight_removal"] = _merge_preset(deepcopy(base["highlight_removal"]), INTENSITY_PATH, rm)
    return cfg


def cfg_keyregion() -> dict:
    cfg = cfg_base()
    cfg["regions"]["key_region_detection_mode"] = True
    return cfg


def cfg_no_extreme() -> dict:
    cfg = cfg_base()
    cfg["highlight_removal"]["extreme_core_extra_l"] = 999
    return cfg


def cfg_chroma_up() -> dict:
    cfg = cfg_base()
    cfg["highlight_removal"]["chroma_restore_strength"] = 0.55
    cfg["highlight_removal"]["brightness_suppress_strength"] = 0.80
    return cfg


def cfg_fullres() -> dict:
    cfg = cfg_base()
    cfg["pipeline"]["process_scale"] = 1.0
    return cfg


def _cfg_extreme(extra_l: float) -> dict:
    cfg = cfg_base()
    cfg["highlight_removal"]["extreme_core_extra_l"] = extra_l
    return cfg


def _cfg_better() -> dict:
    from highlight_removal.utils import load_yaml
    from common import ROOT

    cfg = cfg_base()
    overlay = load_yaml(ROOT / "configs" / "experimental_better.yaml")
    cfg["highlight_removal"].update(overlay.get("highlight_removal") or {})
    return cfg


def _cfg_no_extreme_strong() -> dict:
    cfg = cfg_no_extreme()
    cfg["highlight_removal"]["brightness_suppress_strength"] = 1.0
    cfg["highlight_removal"]["final_blend_alpha"] = 1.0
    cfg["highlight_removal"]["faithful_luminance_floor"] = 0.60
    return cfg


# ------------------------------------------------------------ 算法级变体

def drm_minrgb_removal(image_bgr, hard, soft, regions, params) -> RemovalOutput:
    """经典双色反射：镜面分量近似 minRGB 超出邻域皮肤基线的部分，逐像素减除。

    params["drm_spec_scale"]（默认 1.0）控制减除强度 λ。
    """
    scale = float(params.get("drm_spec_scale", 1.0))
    treatable = mask_to_uint8(regions.masks.get("treatable_skin", regions.masks["skin"])) > 0
    protect = mask_to_uint8(regions.masks["protect"]) > 0
    alpha = normalize_mask(soft)
    alpha[(~treatable) | protect] = 0.0

    img = image_bgr.astype(np.float32)
    minc = img.min(axis=2)
    # 邻域正常皮肤（不含 hard 区）minRGB 基线。
    normal = treatable & (mask_to_uint8(hard) == 0)
    nf = normal.astype(np.float32)
    sigma = 24.0
    num = cv2.GaussianBlur(minc * nf, (0, 0), sigma)
    den = cv2.GaussianBlur(nf, (0, 0), sigma)
    diffuse_min = num / np.maximum(den, 1e-3)
    spec = np.clip(minc - diffuse_min, 0, None) * alpha * scale
    out = np.clip(img - spec[..., None], 0, 255).astype(np.uint8)
    keep = (alpha <= 0.001)[..., None]
    result = np.where(keep, image_bgr, out).astype(np.uint8)
    diff = cv2.absdiff(image_bgr, result)
    return RemovalOutput(result, mask_to_uint8(hard), np.clip(diff.astype(np.float32) * 4, 0, 255).astype(np.uint8), [])


def make_drm_removal(scale: float):
    def fn(image_bgr, hard, soft, regions, params):
        p = dict(params)
        p["drm_spec_scale"] = scale
        return drm_minrgb_removal(image_bgr, hard, soft, regions, p)
    return fn


def make_drm_lab(scale: float = 0.7, chroma_strength: float = 0.85):
    """修正版 DRM：目检发现直接 RGB 减除会在高光核心留下灰蓝斑
    （核心像素本来就发白，减亮度后色度缺失）。改为 Lab 域：
    - L 通道减去 λ·alpha·spec（spec 仍来自 minRGB 超出邻域基线）；
    - a/b 通道按 spec 强度向邻域正常皮肤基线回归（色度恢复）；
    - L 不允许低于邻域皮肤基线 - 2，防止过暗。"""
    def fn(image_bgr, hard, soft, regions, params):
        treatable = mask_to_uint8(regions.masks.get("treatable_skin", regions.masks["skin"])) > 0
        protect = mask_to_uint8(regions.masks["protect"]) > 0
        alpha = normalize_mask(soft)
        alpha[(~treatable) | protect] = 0.0

        img = image_bgr.astype(np.float32)
        minc = img.min(axis=2)
        normal = treatable & (mask_to_uint8(hard) == 0)
        nf = normal.astype(np.float32)
        sigma = 24.0

        def skin_base(v):
            return cv2.GaussianBlur(v * nf, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(nf, (0, 0), sigma), 1e-3)

        spec = np.clip(minc - skin_base(minc), 0, None)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        L_base = skin_base(L)
        a_base = skin_base(a)
        b_base = skin_base(b)

        L_new = L - scale * alpha * spec
        L_new = np.maximum(L_new, np.minimum(L, L_base - 2.0))
        w = np.clip(alpha * spec / 12.0, 0, 1) * float(chroma_strength)
        a_new = a + w * (a_base - a)
        b_new = b + w * (b_base - b)
        out_lab = np.stack([np.clip(L_new, 0, 255), np.clip(a_new, 0, 255), np.clip(b_new, 0, 255)], axis=2)
        out = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        keep = (alpha <= 0.001)[..., None]
        result = np.where(keep, image_bgr, out).astype(np.uint8)
        diff = cv2.absdiff(image_bgr, result)
        return RemovalOutput(result, mask_to_uint8(hard), np.clip(diff.astype(np.float32) * 4, 0, 255).astype(np.uint8), [])
    return fn


def make_floorfix(extreme: bool = True, sigma_mult: float = 1.6, texture_factor: float = 0.16, exclude_shine_baseline: bool = True):
    """保真压制修正实验（两个自由度）：
    - exclude_shine_baseline：_faithful_suppress 的 local_skin 亮度基线用普通高斯，
      大片油光会把自身基线抬高，导致 target_L 的下限 floor 绑死、压不下去；
      改用「排除 hard 油光区的 masked Gaussian」皮肤基线。
    - texture_factor：产线硬编码 0.16 的纹理回加系数，hard 区参考图是 Telea 平滑填充，
      0.16*0.70 只留 11% 高频 → 磨皮感；提高该系数验证纹理保留。
    其余逐行同产线。"""
    from highlight_removal.remove_highlight import _edge_protected_alpha, _inpaint_lab_reference, _p, _strong_inpaint

    def faithful_v2(image_bgr, hard_mask, soft_mask, regions, params):
        brightness_strength = float(_p(params, "brightness_suppress_strength", 0.74))
        color_strength = float(_p(params, "chroma_restore_strength", 0.30))
        texture_strength = float(_p(params, "texture_preserve_strength", 0.78))
        edge_strength = float(_p(params, "edge_protect_strength", 0.45))
        final_alpha = float(_p(params, "final_blend_alpha", 0.92))
        radius = int(_p(params, "inpainting_radius", 3))
        luminance_floor = float(_p(params, "faithful_luminance_floor", 0.86))

        alpha0 = _edge_protected_alpha(image_bgr, soft_mask, edge_strength)
        alpha_l = (alpha0 * brightness_strength * final_alpha).clip(0, 1)
        alpha_c = (alpha0 * color_strength * final_alpha).clip(0, 1)

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = _inpaint_lab_reference(image_bgr, hard_mask, radius)
        L = lab[:, :, 0]
        ref_L = ref_lab[:, :, 0]
        low = cv2.GaussianBlur(L, (0, 0), 2.2)
        texture = L - low
        texture_keep = texture * (float(texture_factor) * np.clip(texture_strength, 0, 1))

        if exclude_shine_baseline:
            # 基线来自排除油光核心的正常皮肤。
            skin = mask_to_uint8(regions.masks["skin"]) > 0
            protect = mask_to_uint8(regions.masks["protect"]) > 0
            normal = skin & (~protect) & (mask_to_uint8(hard_mask) == 0)
            nf = normal.astype(np.float32)
            sigma = max(8.0, float(_p(params, "local_sigma", 6.0)) * float(sigma_mult))
            num = cv2.GaussianBlur(L * nf, (0, 0), sigma)
            den = cv2.GaussianBlur(nf, (0, 0), sigma)
            local_skin = num / np.maximum(den, 1e-3)
        else:
            local_skin = cv2.GaussianBlur(L, (0, 0), max(8.0, float(_p(params, "local_sigma", 6.0)) * 1.6))

        min_allowed = np.maximum(L * luminance_floor, local_skin * 0.93)
        target_L = np.maximum(ref_L + texture_keep, min_allowed)
        target_L = np.maximum(target_L, local_skin - 5.0)
        target_L = np.minimum(target_L, L + 0.5)

        out_lab = lab.copy()
        out_lab[:, :, 0] = L * (1 - alpha_l) + target_L * alpha_l
        out_lab[:, :, 1] = lab[:, :, 1] * (1 - alpha_c) + ref_lab[:, :, 1] * alpha_c
        out_lab[:, :, 2] = lab[:, :, 2] * (1 - alpha_c) + ref_lab[:, :, 2] * alpha_c
        out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        keep = (alpha0 <= 0.001)[..., None]
        return np.where(keep, image_bgr, out).astype(np.uint8)

    def fn(image_bgr, hard, soft, regions, params):
        hard_u8 = mask_to_uint8(hard).copy()
        soft_u8 = mask_to_uint8(soft).copy()
        skin = mask_to_uint8(regions.masks["skin"]) > 0
        protect = mask_to_uint8(regions.masks["protect"]) > 0
        hard_u8[(~skin) | protect] = 0
        soft_u8[(~skin) | protect] = 0
        faithful = faithful_v2(image_bgr, hard_u8, soft_u8, regions, params)
        result = faithful
        if extreme:
            L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            extra = float(_p(params, "extreme_core_extra_l", 12))
            lab_l_thr = float(_p(params, "lab_l_threshold", 188))
            extreme_m = ((hard_u8 > 0) & (L >= lab_l_thr + extra)).astype(np.uint8) * 255
            if int((extreme_m > 0).sum()) > 0:
                extreme_soft = cv2.GaussianBlur(extreme_m, (0, 0), max(1.0, float(_p(params, "mask_blur_radius", 9)) / 2.5))
                result = _strong_inpaint(image_bgr, extreme_m, extreme_soft, params, base_image=faithful)
        immutable = ((soft_u8 == 0) | protect | (~skin))[..., None]
        result = np.where(immutable, image_bgr, result).astype(np.uint8)
        diff = cv2.absdiff(image_bgr, result)
        return RemovalOutput(result, hard_u8, np.clip(diff.astype(np.float32) * 4, 0, 255).astype(np.uint8), [])
    return fn


def make_hybrid_drm_faithful(scale: float):
    """先做 λ 缩放的 DRM 物理减除（保纹理），再走保真 Lab 压制（无 extreme Telea）。"""
    def fn(image_bgr, hard, soft, regions, params):
        p = dict(params)
        p["drm_spec_scale"] = scale
        step1 = drm_minrgb_removal(image_bgr, hard, soft, regions, p)
        p2 = dict(params)
        p2["extreme_core_extra_l"] = 999
        step2 = remove_highlight(step1.result_bgr, hard, soft, regions, p2)
        diff = cv2.absdiff(image_bgr, step2.result_bgr)
        return RemovalOutput(
            step2.result_bgr, mask_to_uint8(hard),
            np.clip(diff.astype(np.float32) * 4, 0, 255).astype(np.uint8),
            step1.warnings + step2.warnings,
        )
    return fn


def _guided_lab_reference(image_bgr, hard_mask, radius):
    """把参考图中 bilateral → guided filter（边缘感知），Telea 部分保持不变。"""
    hard = mask_to_uint8(hard_mask)
    radius = int(max(1, min(9, radius)))
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    if int((hard > 0).sum()) == 0:
        return lab.astype(np.float32)
    channels = [cv2.inpaint(c, hard, radius, cv2.INPAINT_TELEA) for c in cv2.split(lab)]
    ref = cv2.merge(channels).astype(np.float32)
    guided = cv2.ximgproc.guidedFilter(guide=image_bgr, src=image_bgr, radius=11, eps=45.0 * 45.0)
    ref2 = cv2.cvtColor(guided, cv2.COLOR_BGR2LAB).astype(np.float32)
    return 0.72 * ref + 0.28 * ref2


def log_shine_masks(image_bgr, hard, soft, regions, hd_params):
    """Y 通道 LoG（亮度局部极大）油光候选，OR 进现有 mask。"""
    treatable = mask_to_uint8(regions.masks.get("treatable_skin", regions.masks["skin"])) > 0
    protect = mask_to_uint8(regions.masks["protect"]) > 0
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), 3.0)
    log_resp = -cv2.Laplacian(blur, cv2.CV_32F, ksize=5)  # 正值 = 亮度局部凸起
    region = treatable & (~protect)
    if int(region.sum()) < 64:
        return hard, soft
    thr = float(np.percentile(log_resp[region], 90))
    l_thr = float(np.percentile(L[region], 70))
    cand = region & (log_resp >= max(thr, 8.0)) & (L >= l_thr)
    cand_u8 = cand.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE, k)
    new_hard = cv2.bitwise_or(mask_to_uint8(hard), cand_u8)
    blur_r = int(hd_params.get("mask_blur_radius", 8))
    gain = float(hd_params.get("soft_mask_gain", 1.85))
    extra_soft = cv2.GaussianBlur(cand_u8, (2 * blur_r + 1, 2 * blur_r + 1), 0)
    extra_soft = np.clip(extra_soft.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    new_soft = np.maximum(mask_to_uint8(soft), extra_soft)
    new_hard[protect] = 0
    new_soft[protect] = 0
    return new_hard, new_soft


# ---------------------------------------------------------------- 运行器

def run_config_variant(img, face, cfg) -> dict:
    out = _process_core(img, face, [face], cfg)
    return {"result": out.result_bgr, "metrics": out.metrics, "warnings": out.warnings}


def run_custom_variant(img, face, cfg, *, removal_fn=None, mask_fn=None, patch_ref=False) -> dict:
    highlight, hard, soft, _, _, regions = _detect_highlight_masks(img, face, cfg)
    if mask_fn is not None:
        hard, soft = mask_fn(img, hard, soft, regions, cfg.get("highlight_detection", {}))
    params = _removal_params(cfg)
    if removal_fn is not None:
        removal = removal_fn(img, hard, soft, regions, params)
    elif patch_ref:
        orig = rh_mod._inpaint_lab_reference
        rh_mod._inpaint_lab_reference = _guided_lab_reference
        try:
            removal = remove_highlight(img, hard, soft, regions, params)
        finally:
            rh_mod._inpaint_lab_reference = orig
    else:
        removal = remove_highlight(img, hard, soft, regions, params)
    metrics, metric_warnings = compute_metrics(img, removal.result_bgr, removal.applied_mask, regions, face, cfg.get("highlight_removal", {}))
    warnings = [w for w in (regions.warnings + highlight.warnings + removal.warnings + metric_warnings) if w]
    return {"result": removal.result_bgr, "metrics": metrics, "warnings": list(dict.fromkeys(warnings))}


VARIANTS: Dict[str, Callable[[], dict]] = {}


def _register_variants():
    VARIANTS["base"] = lambda: {"cfg": cfg_base()}
    VARIANTS["keyregion"] = lambda: {"cfg": cfg_keyregion()}
    for det in ("灵敏", "正常", "弱"):
        for rm in ("强力", "正常", "削弱"):
            if det == "灵敏" and rm == "强力":
                continue  # 即 base
            VARIANTS[f"det{det}_rm{rm}"] = lambda det=det, rm=rm: {"cfg": cfg_grid(det, rm)}
    VARIANTS["no_extreme"] = lambda: {"cfg": cfg_no_extreme()}
    VARIANTS["chroma_up"] = lambda: {"cfg": cfg_chroma_up()}
    VARIANTS["fullres"] = lambda: {"cfg": cfg_fullres()}
    VARIANTS["drm_minrgb"] = lambda: {"cfg": cfg_base(), "removal_fn": drm_minrgb_removal}
    VARIANTS["guided_filter"] = lambda: {"cfg": cfg_base(), "patch_ref": True}
    VARIANTS["log_shine_or"] = lambda: {"cfg": cfg_base(), "mask_fn": log_shine_masks}
    # 组合变体（第二轮）
    VARIANTS["drm06"] = lambda: {"cfg": cfg_base(), "removal_fn": make_drm_removal(0.6)}
    VARIANTS["hybrid_drm_faithful"] = lambda: {"cfg": cfg_base(), "removal_fn": make_hybrid_drm_faithful(0.55)}
    VARIANTS["no_extreme_strong"] = lambda: {"cfg": _cfg_no_extreme_strong()}
    # 第三轮：Lab 域 DRM（L 压制 + 色度恢复）
    VARIANTS["drm_lab"] = lambda: {"cfg": cfg_base(), "removal_fn": make_drm_lab(0.7, 0.85)}
    VARIANTS["drm_lab_mild"] = lambda: {"cfg": cfg_base(), "removal_fn": make_drm_lab(0.55, 0.7)}
    # 第四轮：保真压制的皮肤亮度基线排除油光区（floor 修正）
    VARIANTS["floorfix"] = lambda: {"cfg": cfg_base(), "removal_fn": make_floorfix(extreme=True)}
    VARIANTS["floorfix_ne"] = lambda: {"cfg": cfg_base(), "removal_fn": make_floorfix(extreme=False)}
    # 第五轮：hard 区纹理回加系数 0.16 → 0.35（抗磨皮）
    VARIANTS["texup"] = lambda: {"cfg": cfg_base(), "removal_fn": make_floorfix(extreme=True, texture_factor=0.35, exclude_shine_baseline=False)}
    VARIANTS["texup_floorfix"] = lambda: {"cfg": cfg_base(), "removal_fn": make_floorfix(extreme=True, texture_factor=0.35, exclude_shine_baseline=True)}
    # 第六轮：自然端组合（floorfix+纹理提升、无 extreme）与 extreme 阈值收紧（纯配置）
    VARIANTS["texup_ne"] = lambda: {"cfg": cfg_base(), "removal_fn": make_floorfix(extreme=False, texture_factor=0.35, exclude_shine_baseline=True)}
    VARIANTS["extreme30"] = lambda: {"cfg": _cfg_extreme(30)}
    # 第七轮：平衡组合（extreme 收紧到近饱和 + 纹理回加 0.35 + floor 修正）
    VARIANTS["balanced"] = lambda: {"cfg": _cfg_extreme(30), "removal_fn": make_floorfix(extreme=True, texture_factor=0.35, exclude_shine_baseline=True)}
    # 同一组合改经 configs/experimental_better.yaml 的生产配置开关（验证开关实现）
    VARIANTS["balanced_cfg"] = lambda: {"cfg": _cfg_better()}


_register_variants()


def fixed_eval_masks(name: str):
    d = np.load(OUT_DIR / "audit" / f"{name}_masks.npz")
    return d["treatable"] > 0, d["protect"] > 0, d["face_mask"] > 0


def shine_zone(img, treatable) -> np.ndarray:
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    if int(treatable.sum()) < 64:
        return np.zeros_like(treatable)
    thr = float(np.percentile(L[treatable], 90))
    return treatable & (L >= thr)


def evaluate(name, img, result, metrics, warnings) -> dict:
    treatable, protect, face_mask = fixed_eval_masks(name)
    zone = shine_zone(img, treatable)
    zone_u8 = zone.astype(np.uint8) * 255
    tr_u8 = treatable.astype(np.uint8) * 255
    rs_post = residual_shine(result, zone_u8, tr_u8)

    diff = cv2.absdiff(img, result)
    dg = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed = (dg > 2) & face_mask
    lab0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab1 - lab0
    if changed.any():
        m_dl = float(np.abs(delta[:, :, 0][changed]).mean())
        m_da = float(np.abs(delta[:, :, 1][changed]).mean())
        m_db = float(np.abs(delta[:, :, 2][changed]).mean())
    else:
        m_dl = m_da = m_db = 0.0

    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    lv0 = lap_var(g0, zone_u8)
    lv1 = lap_var(g1, zone_u8)

    protect_touch = bool(
        metrics.get("眼睛区域是否被修改") or metrics.get("嘴唇区域是否被修改") or metrics.get("眉毛区域是否被修改")
    )
    return {
        "残留dL": rs_post["residual_dl"],
        "改动占脸": round(float(changed.sum() / max(1, face_mask.sum())), 4),
        "m|dL|": round(m_dl, 3),
        "m|da|": round(m_da, 3),
        "m|db|": round(m_db, 3),
        "纹理比": round(lv1 / lv0, 4) if lv0 > 0 else 1.0,
        "SSIM": round(masked_ssim(img, result, tr_u8), 4),
        "保护区被改": protect_touch,
        "背景被改": bool(metrics.get("背景是否被修改", False)),
        "警告数": len(warnings),
    }


def main(argv) -> int:
    wanted = argv or list(VARIANTS.keys())
    all_rows: Dict[str, Any] = {}
    images = {n: imread(n) for n in NAMES}
    faces = {n: load_face(n) for n in NAMES}

    # 处理前基线数字（所有变体共享）
    pre = {}
    for n in NAMES:
        treatable, _, _ = fixed_eval_masks(n)
        zone = shine_zone(images[n], treatable)
        pre[n] = residual_shine(images[n], zone.astype(np.uint8) * 255, treatable.astype(np.uint8) * 255)["residual_dl"]
    all_rows["_处理前残留dL"] = pre

    for vname in wanted:
        spec = VARIANTS[vname]()
        cfg = spec.pop("cfg")
        rows = {}
        vdir = VAR_DIR / vname
        vdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for n in NAMES:
            if spec:
                r = run_custom_variant(images[n], faces[n], cfg, **spec)
            else:
                r = run_config_variant(images[n], faces[n], cfg)
            rows[n] = evaluate(n, images[n], r["result"], r["metrics"], r["warnings"])
            rows[n]["警告"] = r["warnings"]
            cv2.imwrite(str(vdir / f"{n}.png"), r["result"])
        elapsed = time.perf_counter() - t0

        agg = {
            "平均残留dL": round(float(np.mean([rows[n]["残留dL"] for n in NAMES])), 3),
            "平均改动占脸": round(float(np.mean([rows[n]["改动占脸"] for n in NAMES])), 4),
            "平均m|dL|": round(float(np.mean([rows[n]["m|dL|"] for n in NAMES])), 3),
            "平均m|da|": round(float(np.mean([rows[n]["m|da|"] for n in NAMES])), 3),
            "平均m|db|": round(float(np.mean([rows[n]["m|db|"] for n in NAMES])), 3),
            "平均纹理比": round(float(np.mean([rows[n]["纹理比"] for n in NAMES])), 4),
            "平均SSIM": round(float(np.mean([rows[n]["SSIM"] for n in NAMES])), 4),
            "保护区被改张数": int(sum(rows[n]["保护区被改"] for n in NAMES)),
            "背景被改张数": int(sum(rows[n]["背景被改"] for n in NAMES)),
            "总警告数": int(sum(rows[n]["警告数"] for n in NAMES)),
            "总耗时s": round(elapsed, 1),
        }
        all_rows[vname] = {"汇总": agg, "逐张": rows}
        print(f"== {vname}: 残留dL={agg['平均残留dL']} 纹理比={agg['平均纹理比']} "
              f"m|da|={agg['平均m|da|']} m|db|={agg['平均m|db|']} SSIM={agg['平均SSIM']} "
              f"警告={agg['总警告数']} 保护区={agg['保护区被改张数']} 耗时={agg['总耗时s']}s")

    dump_json(OUT_DIR / "experiments.json", all_rows)
    print(f"\n实验完成，结果写入 {OUT_DIR / 'experiments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
