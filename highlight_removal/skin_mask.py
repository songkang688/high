from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from .utils import blur_mask, mask_to_uint8, morph


def estimate_skin_mask(image_bgr: np.ndarray, face_mask: np.ndarray, protect_mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """基于面部轮廓和颜色自适应估计皮肤 mask。

    这里的基础约束是关键点生成的 face_mask，颜色模型只用于剔除明显非皮肤暗区域；
    高光像素由于可能低饱和/高亮，仍保留在皮肤候选内，避免漏检。
    """
    face = mask_to_uint8(face_mask) > 0
    protect = mask_to_uint8(protect_mask) > 0
    base = face & (~protect)
    if base.sum() == 0:
        return np.zeros(image_bgr.shape[:2], np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    sample = base & (v > 40) & (v < 245) & (s > 8)
    if sample.sum() < 50:
        sample = base & (v > 25)

    med_cr = np.median(cr[sample]) if sample.sum() > 0 else 150
    med_cb = np.median(cb[sample]) if sample.sum() > 0 else 110
    med_a = np.median(a[sample]) if sample.sum() > 0 else 135
    med_b = np.median(b[sample]) if sample.sum() > 0 else 135

    # 宽松通用肤色范围 + 个体自适应范围。
    generic = (cr >= 125) & (cr <= 185) & (cb >= 70) & (cb <= 145) & (v >= 35)
    adaptive = (np.abs(cr - med_cr) <= 34) & (np.abs(cb - med_cb) <= 34) & (np.abs(a - med_a) <= 28) & (np.abs(b - med_b) <= 34) & (v >= 35)
    # 高光不一定是纯白低饱和。额头/鼻梁常见暖黄色油光，S 可到 140~170，必须保留。
    bright_highlight_like = (v >= 165) & (s <= 175)  # 高光经常高亮，不能因肤色阈值被排除。
    dark_non_skin = (v < 45) | ((v < 85) & (s > 70))

    skin = base & ((generic | adaptive | bright_highlight_like) & (~dark_non_skin))
    # 关键点轮廓已经很可靠，颜色阈值只做软约束；对断裂区域进行闭运算。
    smooth_radius = int(params.get("skin_mask_smooth_radius", 3))
    skin_u8 = (skin.astype(np.uint8) * 255)
    skin_u8 = morph(skin_u8, "close", max(1, smooth_radius))
    skin_u8 = morph(skin_u8, "open", 1)
    if smooth_radius > 0:
        skin_u8 = blur_mask(skin_u8, smooth_radius)
        skin_u8 = ((skin_u8 > 80).astype(np.uint8) * 255)
    skin_u8[protect] = 0
    return skin_u8
