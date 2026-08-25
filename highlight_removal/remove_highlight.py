from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

from .face_regions import RegionMasks
from .utils import mask_to_uint8, normalize_mask


@dataclass
class RemovalOutput:
    result_bgr: np.ndarray
    applied_mask: np.ndarray
    diff_bgr: np.ndarray
    warnings: List[str]


def _p(params: Dict[str, Any], names, default):
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in params:
            return params[n]
    return default


def _edge_protected_alpha(image_bgr: np.ndarray, soft_mask: np.ndarray, edge_strength: float) -> np.ndarray:
    alpha = normalize_mask(soft_mask)
    if edge_strength <= 0:
        return alpha
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 130)
    edges = cv2.GaussianBlur(edges, (5, 5), 0).astype(np.float32) / 255.0
    return (alpha * (1.0 - np.clip(edge_strength, 0, 1) * edges)).clip(0, 1)


def _inpaint_lab_reference(image_bgr: np.ndarray, hard_mask: np.ndarray, radius: int) -> np.ndarray:
    hard = mask_to_uint8(hard_mask)
    radius = int(max(1, min(9, radius)))
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    if int((hard > 0).sum()) == 0:
        return lab.astype(np.float32)
    # 分通道 inpaint 比直接 BGR inpaint 更稳定，能减少灰斑。
    channels = []
    for c in cv2.split(lab):
        channels.append(cv2.inpaint(c, hard, radius, cv2.INPAINT_TELEA))
    ref = cv2.merge(channels).astype(np.float32)
    # 再用双边滤波提供柔和肤色基线。
    bilateral = cv2.bilateralFilter(image_bgr, d=11, sigmaColor=45, sigmaSpace=45)
    ref2 = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB).astype(np.float32)
    return 0.72 * ref + 0.28 * ref2


def _faithful_suppress(image_bgr: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
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
    # 保留高频纹理：目标亮度来自 inpaint 参考，但局部纹理部分从原图回加少量。
    # texture_keep_factor 默认 0.16（与历史行为逐位一致）；实验配置可调高以抗磨皮。
    low = cv2.GaussianBlur(L, (0, 0), 2.2)
    texture = L - low
    texture_keep = texture * (float(_p(params, "texture_keep_factor", 0.16)) * np.clip(texture_strength, 0, 1))

    # 只降低亮度，不把正常皮肤拉得过暗；同时允许明显白斑向周围肤色回归。
    baseline_sigma = max(8.0, float(_p(params, "local_sigma", 6.0)) * 1.6)
    valid_mask = params.get("_faithful_valid_mask")
    if valid_mask is not None and bool(_p(params, "faithful_baseline_exclude_shine", False)):
        # 实验开关（默认关闭）：皮肤亮度基线排除油光核心，防止大片油光把
        # 自身基线抬高、target_L 的下限 floor 绑死导致压不下去。
        nf = (valid_mask & (mask_to_uint8(hard_mask) == 0)).astype(np.float32)
        num = cv2.GaussianBlur(L * nf, (0, 0), baseline_sigma)
        den = cv2.GaussianBlur(nf, (0, 0), baseline_sigma)
        local_skin = num / np.maximum(den, 1e-3)
    else:
        local_skin = cv2.GaussianBlur(L, (0, 0), baseline_sigma)
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


def _strong_inpaint(image_bgr: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray, params: Dict[str, Any], base_image: np.ndarray | None = None) -> np.ndarray:
    source = image_bgr if base_image is None else base_image
    radius = int(max(1, min(9, int(_p(params, "inpainting_radius", 3)))))
    alpha_strength = float(_p(params, "poisson_alpha_strength", 0.42))
    final_alpha = float(_p(params, "final_blend_alpha", 0.92))
    hard = mask_to_uint8(hard_mask)
    if int((hard > 0).sum()) == 0:
        return source.copy()
    inpainted = cv2.inpaint(source, hard, radius, cv2.INPAINT_TELEA)
    # 防止强修复把细节全部抹平：再和保真结果/原图有限融合。
    alpha = (normalize_mask(soft_mask) * alpha_strength * final_alpha).clip(0, 1)[..., None]
    out = source.astype(np.float32) * (1 - alpha) + inpainted.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def remove_highlight(image_bgr: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray, regions: RegionMasks, params: Dict[str, Any]) -> RemovalOutput:
    enable_roi = bool(_p(params, "enable_roi_removal", True))
    padding_ratio = float(_p(params, "roi_padding_ratio", 0.12))
    if enable_roi:
        from .utils import compute_roi_bbox

        roi = compute_roi_bbox(hard_mask, soft_mask, image_bgr.shape, padding_ratio)
        if roi is not None:
            x, y, rw, rh = roi
            if rw < image_bgr.shape[1] or rh < image_bgr.shape[0]:
                cropped_regions = RegionMasks(
                    {k: v[y : y + rh, x : x + rw] for k, v in regions.masks.items()},
                    regions.geometry,
                    regions.warnings,
                )
                out = _remove_highlight_core(
                    image_bgr[y : y + rh, x : x + rw],
                    hard_mask[y : y + rh, x : x + rw],
                    soft_mask[y : y + rh, x : x + rw],
                    cropped_regions,
                    params,
                )
                result = image_bgr.copy()
                result[y : y + rh, x : x + rw] = out.result_bgr
                applied_full = mask_to_uint8(hard_mask).copy()
                diff = cv2.absdiff(image_bgr, result)
                diff_vis = np.clip(diff.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
                return RemovalOutput(result_bgr=result, applied_mask=applied_full, diff_bgr=diff_vis, warnings=out.warnings)
    return _remove_highlight_core(image_bgr, hard_mask, soft_mask, regions, params)


def _remove_highlight_core(image_bgr: np.ndarray, hard_mask: np.ndarray, soft_mask: np.ndarray, regions: RegionMasks, params: Dict[str, Any]) -> RemovalOutput:
    warnings: List[str] = []
    mode = str(_p(params, "mode", "保真"))
    skin = mask_to_uint8(regions.masks["skin"]) > 0
    protect = mask_to_uint8(regions.masks["protect"]) > 0
    hard = mask_to_uint8(hard_mask).copy()
    soft = mask_to_uint8(soft_mask).copy()
    hard[(~skin) | protect] = 0
    soft[(~skin) | protect] = 0

    if bool(_p(params, "faithful_baseline_exclude_shine", False)):
        params = dict(params)
        params["_faithful_valid_mask"] = skin & (~protect)

    face_area = max(1, int((mask_to_uint8(regions.masks["face_mask"]) > 0).sum()))
    mod_area_ratio = float((hard > 0).sum() / face_area)
    max_allowed = float(_p(params, "max_allowed_modify_area_ratio", 0.15))
    if mod_area_ratio > max_allowed:
        warnings.append("当前修改区域过大，可能影响真实性，请降低处理强度")

    if mode == "强修复":
        result = _strong_inpaint(image_bgr, hard, soft, params)
    elif mode == "混合":
        faithful = _faithful_suppress(image_bgr, hard, soft, params)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        # 只对极端核心二次 inpaint，普通油光仍保持保真 L 通道修复。
        extra = float(_p(params, "extreme_core_extra_l", 12))
        lab_l_thr = float(_p(params, "lab_l_threshold", 188))
        extreme = ((hard > 0) & (L >= lab_l_thr + extra)).astype(np.uint8) * 255
        if int((extreme > 0).sum()) > 0:
            extreme_soft = cv2.GaussianBlur(extreme, (0, 0), max(1.0, float(_p(params, "mask_blur_radius", 9)) / 2.5))
            result = _strong_inpaint(image_bgr, extreme, extreme_soft, params, base_image=faithful)
        else:
            result = faithful
    else:
        result = _faithful_suppress(image_bgr, hard, soft, params)

    # 强制保护：五官、背景、非皮肤区域完全回写原图。
    immutable = ((mask_to_uint8(soft) == 0) | protect | (~skin))[..., None]
    result = np.where(immutable, image_bgr, result).astype(np.uint8)

    diff = cv2.absdiff(image_bgr, result)
    diff_vis = np.clip(diff.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    return RemovalOutput(result_bgr=result, applied_mask=hard, diff_bgr=diff_vis, warnings=warnings)
