from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .face_detect import FaceResult
from .face_regions import RegionMasks
from .utils import mask_to_uint8


def _ratio(mask: np.ndarray, denom_mask: np.ndarray) -> float:
    denom = max(1, int((mask_to_uint8(denom_mask) > 0).sum()))
    return float((mask_to_uint8(mask) > 0).sum() / denom)


def _modified_in(mask: np.ndarray, diff_gray: np.ndarray, threshold: int = 2) -> bool:
    m = mask_to_uint8(mask) > 0
    if m.sum() == 0:
        return False
    return bool(np.any(diff_gray[m] > threshold))


def compute_metrics(
    original_bgr: np.ndarray,
    result_bgr: np.ndarray,
    hard_mask: np.ndarray,
    regions: RegionMasks,
    face: FaceResult,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    face_mask = mask_to_uint8(regions.masks["face_mask"]) > 0
    skin = mask_to_uint8(regions.masks["skin"]) > 0
    protect = mask_to_uint8(regions.masks["protect"]) > 0
    diff = cv2.absdiff(original_bgr, result_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed = diff_gray > 2

    lab0 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab1 - lab0
    dL = delta[:, :, 0]
    dE = np.sqrt((delta ** 2).sum(axis=2))

    mod_face = changed & face_mask
    metrics: Dict[str, Any] = {
        "人脸检测置信度": round(float(face.confidence), 4),
        "关键点数量": int(0 if face.landmarks is None else len(face.landmarks)),
        "人脸 bounding box 坐标": [int(v) for v in face.bbox],
        "高光 mask 面积占人脸比例": round(_ratio(hard_mask, regions.masks["face_mask"]), 6),
        "实际修改像素占人脸比例": round(float(mod_face.sum() / max(1, face_mask.sum())), 6),
        "平均亮度变化": round(float(np.mean(np.abs(dL[mod_face]))) if mod_face.any() else 0.0, 4),
        "最大亮度变化": round(float(np.max(np.abs(dL[mod_face]))) if mod_face.any() else 0.0, 4),
        "平均色差变化": round(float(np.mean(dE[mod_face])) if mod_face.any() else 0.0, 4),
        "最大色差变化": round(float(np.max(dE[mod_face])) if mod_face.any() else 0.0, 4),
        "眼睛区域是否被修改": _modified_in(cv2.bitwise_or(regions.masks["left_eye"], regions.masks["right_eye"]), diff_gray),
        "嘴唇区域是否被修改": _modified_in(regions.masks["lips"], diff_gray),
        "眉毛区域是否被修改": _modified_in(
            cv2.bitwise_or(regions.masks["left_brow_protect"], regions.masks["right_brow_protect"]),
            diff_gray,
        ),
        "背景是否被修改": bool(np.any(changed & (~face_mask))),
    }

    max_area = float(params.get("max_allowed_modify_area_ratio", 0.15))
    max_mean_L = float(params.get("max_allowed_mean_brightness_change", 18))
    max_mean_E = float(params.get("max_allowed_local_color_delta", 18))
    if metrics["实际修改像素占人脸比例"] > max_area:
        warnings.append("修改区域超过人脸皮肤区域的 15% 或当前设置上限")
    if metrics["眼睛区域是否被修改"]:
        warnings.append("眼睛区域被修改，已触发真实性警告")
    if metrics["嘴唇区域是否被修改"]:
        warnings.append("嘴唇区域被修改，已触发真实性警告")
    if metrics["眉毛区域是否被修改"]:
        warnings.append("眉毛区域被修改，已触发真实性警告")
    if metrics["背景是否被修改"]:
        warnings.append("背景区域被修改，已触发真实性警告")
    if metrics["平均亮度变化"] > max_mean_L:
        warnings.append("平均亮度变化过大，可能影响真实性，请降低亮度压制强度")
    if metrics["平均色差变化"] > max_mean_E:
        warnings.append("平均色差变化过大，可能导致肤色失真，请降低色度恢复或融合强度")

    # 肤色偏移审计：只在实际修改区域检测平均 a/b 偏移。
    if mod_face.any():
        mean_da = float(np.mean(delta[:, :, 1][mod_face]))
        mean_db = float(np.mean(delta[:, :, 2][mod_face]))
    else:
        mean_da = 0.0
        mean_db = 0.0
    metrics["平均 a 通道变化"] = round(mean_da, 4)
    metrics["平均 b 通道变化"] = round(mean_db, 4)
    if abs(mean_da) > 6 or abs(mean_db) > 8:
        warnings.append("处理后肤色可能偏灰、偏红或偏黄，请降低色度恢复强度")

    metrics["是否触发真实性警告"] = bool(warnings)
    return metrics, warnings


def metrics_to_text(metrics: Dict[str, Any], warnings: List[str]) -> str:
    lines = []
    for k, v in metrics.items():
        lines.append(f"{k}：{v}")
    if warnings:
        lines.append("\n真实性警告：")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("\n真实性警告：无")
    return "\n".join(lines)
