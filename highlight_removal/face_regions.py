from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

from .face_landmarks import (
    CHIN_AREA,
    FACE_OVAL,
    INNER_LIPS,
    LEFT_BROW,
    LEFT_EYE,
    LEFT_LOWER_EYE,
    MOUTH_CORNERS,
    NOSE_BRIDGE,
    NOSE_TIP,
    NOSE_WING,
    OUTER_LIPS,
    RIGHT_BROW,
    RIGHT_EYE,
    RIGHT_LOWER_EYE,
    _filter_side_brow_pts,
    estimate_head_yaw_ratio,
    get_geometry,
    local_coordinate_grid,
    project_points,
    pts,
)
from .skin_mask import estimate_skin_mask
from .utils import ellipse_mask, fill_hull_mask, fill_poly_mask, line_mask, mask_to_uint8, morph


@dataclass
class RegionMasks:
    masks: Dict[str, np.ndarray]
    geometry: Any
    warnings: List[str]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.masks[key]


def _protect_feature(shape, landmarks, indices, dilate_px: int) -> np.ndarray:
    mask = fill_hull_mask(shape, pts(landmarks, indices))
    if dilate_px > 0:
        mask = morph(mask, "dilate", dilate_px)
    return mask


def _clip_to_face(mask: np.ndarray, face_mask_raw: np.ndarray, protect: np.ndarray) -> np.ndarray:
    out = (
        (mask_to_uint8(mask) > 0)
        & (mask_to_uint8(face_mask_raw) > 0)
        & (mask_to_uint8(protect) == 0)
    )
    return out.astype(np.uint8) * 255


def _clip_to_skin(mask: np.ndarray, skin: np.ndarray, protect: np.ndarray) -> np.ndarray:
    out = (mask_to_uint8(mask) > 0) & (mask_to_uint8(skin) > 0) & (mask_to_uint8(protect) == 0)
    return out.astype(np.uint8) * 255


def _philtrum_box(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    io: float,
    width_ratio: float,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    upper_lip = pts(landmarks, [61, 0, 37, 39, 40, 185, 291, 375, 321, 405, 314, 17, 267, 269, 270])
    nose_bottom = pts(landmarks, [2, 94, 19, 4, 5, 98, 327, 326, 49, 279])
    if len(upper_lip) < 2 or len(nose_bottom) < 2:
        return None
    lip_top_y = float(np.min(upper_lip[:, 1]))
    nose_base_y = float(np.max(nose_bottom[:, 1]))
    if lip_top_y <= nose_base_y + 1:
        return None
    cx = float(np.mean(upper_lip[:, 0]))
    half_w = max(2.0, io * float(width_ratio))
    y0 = int(max(0, np.floor(nose_base_y - io * 0.02)))
    y1 = int(min(h, np.ceil(lip_top_y + io * 0.04)))
    x0 = int(max(0, np.floor(cx - half_w)))
    x1 = int(min(w, np.ceil(cx + half_w)))
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _carve_philtrum_from_protect(
    protect: np.ndarray,
    landmarks: np.ndarray,
    io: float,
    width_ratio: float = 0.16,
) -> np.ndarray:
    """人中不在保护名单内；去掉嘴唇 mask 向上膨胀后误盖住的鼻下区域。"""
    box = _philtrum_box(protect.shape[:2], landmarks, io, width_ratio)
    if box is None:
        return protect
    y0, y1, x0, x1 = box
    out = protect.copy()
    out[y0:y1, x0:x1] = 0
    return out


def _build_philtrum_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    io: float,
    params: Dict[str, Any],
) -> np.ndarray:
    width_ratio = float(params.get("philtrum_width_ratio", 0.16))
    box = _philtrum_box(shape, landmarks, io, width_ratio * 1.08)
    if box is None:
        return np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = box
    mask = np.zeros(shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _smooth_brow_arc_curve(
    shape: tuple[int, int],
    brow_pts: np.ndarray,
    io: float,
    width_ratio: float = 1.0,
    y_shift_ratio: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """MediaPipe 眉点 → 二次曲线拟合 + 强平滑，得到单条月牙形眉弧中心线。"""
    h, w = shape
    if len(brow_pts) < 2:
        empty = np.full(w, h, dtype=np.float32)
        return empty, 0.0, float(w - 1)
    order = np.argsort(brow_pts[:, 0])
    bx = brow_pts[order, 0].astype(np.float64)
    by = brow_pts[order, 1].astype(np.float64)
    xx = np.arange(w, dtype=np.float32)
    if len(bx) >= 3:
        y = np.polyval(np.polyfit(bx, by, 2), xx.astype(np.float64)).astype(np.float32)
    else:
        y = np.interp(xx, bx.astype(np.float32), by.astype(np.float32)).astype(np.float32)
    k = int(max(11, round(io * 0.34)))
    if k % 2 == 0:
        k += 1
    y = cv2.GaussianBlur(y.reshape(1, -1), (k, 1), 0).reshape(-1)
    y -= io * max(0.0, float(y_shift_ratio))
    ext = io * 0.028 * max(0.5, float(width_ratio))
    x_left = float(bx[0]) - ext
    x_right = float(bx[-1]) + ext
    return y, x_left, x_right


def _arc_band_mask(
    shape: tuple[int, int],
    y_top_1d: np.ndarray,
    y_bot_1d: np.ndarray,
    x_left: float,
    x_right: float,
) -> np.ndarray:
    h, w = shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    y_top = np.broadcast_to(y_top_1d, (h, w))
    y_bot = np.broadcast_to(y_bot_1d, (h, w))
    in_x = (xx >= x_left) & (xx <= x_right)
    return (in_x & (yy >= y_top) & (yy <= y_bot)).astype(np.uint8) * 255


def _brow_protect_thickness(
    io: float,
    dilate_px: int,
    shrink_px: int,
    height_ratio: float = 1.0,
) -> tuple[float, float]:
    hr = max(0.5, float(height_ratio))
    pad_up = io * 0.016 * hr
    pad_down = io * 0.009 * hr
    pad_down += max(0, dilate_px - shrink_px) * io * 0.004 * hr
    return pad_up, pad_down


def _build_brow_protect_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    brow_indices,
    eye_indices,
    io: float,
    dilate_px: int,
    shrink_px: int,
    height_ratio: float = 1.0,
    width_ratio: float = 1.0,
    y_shift_ratio: float = 0.0,
) -> np.ndarray:
    """平滑月牙眉带：上厚下薄，沿二次拟合眉弧。"""
    eye_pts = pts(landmarks, eye_indices)
    brow = _filter_side_brow_pts(pts(landmarks, brow_indices), eye_pts, io)
    if len(brow) < 2 or len(eye_pts) < 2:
        return np.zeros(shape, np.uint8)
    eye_span = float(eye_pts[:, 0].max() - eye_pts[:, 0].min())
    if eye_span < io * 0.07:
        return np.zeros(shape, np.uint8)
    y_c, x_l, x_r = _smooth_brow_arc_curve(shape, brow, io, width_ratio, y_shift_ratio)
    pad_up, pad_down = _brow_protect_thickness(io, dilate_px, shrink_px, height_ratio)
    return _arc_band_mask(shape, y_c - pad_up, y_c + pad_down, x_l, x_r)


def _build_residual_brow_ridge_side(
    shape: tuple[int, int],
    brow_pts: np.ndarray,
    eye_pts: np.ndarray,
    io: float,
    face_mask: np.ndarray,
    skin: np.ndarray,
    occupied: np.ndarray,
    side_u_mask: np.ndarray,
    height_ratio: float,
    width_ratio: float,
    dilate_px: int,
    shrink_px: int,
    y_shift_ratio: float = 0.0,
) -> np.ndarray:
    """眉弓 = 眉下～上眼睑之间的剩余皮肤（优先级低于眉/眼/额/颊）。"""
    brow_pts = _filter_side_brow_pts(brow_pts, eye_pts, io)
    if len(brow_pts) < 2 or len(eye_pts) < 2:
        return np.zeros(shape, np.uint8)
    if float(eye_pts[:, 0].max() - eye_pts[:, 0].min()) < io * 0.07:
        return np.zeros(shape, np.uint8)
    h, w = shape
    y_c, x_l, x_r = _smooth_brow_arc_curve(shape, brow_pts, io, width_ratio, y_shift_ratio)
    _, pad_down = _brow_protect_thickness(io, dilate_px, shrink_px, height_ratio)
    y_top = y_c + pad_down
    order_e = np.argsort(eye_pts[:, 0])
    ex, ey = eye_pts[order_e, 0], eye_pts[order_e, 1]
    xx = np.arange(w, dtype=np.float32)
    y_eye = np.interp(xx, ex, ey, left=ey[0], right=ey[-1])
    k = int(max(9, round(io * 0.18)))
    if k % 2 == 0:
        k += 1
    y_eye = cv2.GaussianBlur(y_eye.reshape(1, -1), (k, 1), 0).reshape(-1)
    y_bot = y_eye + io * 0.018
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx_g = np.arange(w, dtype=np.float32)[None, :]
    in_x = (xx_g >= x_l) & (xx_g <= x_r)
    y_top_m = np.broadcast_to(y_top, (h, w))
    y_bot_m = np.broadcast_to(y_bot, (h, w))
    candidate = (
        (face_mask > 0)
        & (skin > 0)
        & side_u_mask
        & in_x
        & (yy >= y_top_m)
        & (yy <= y_bot_m)
    )
    ridge = candidate & (~(occupied > 0))
    return ridge.astype(np.uint8) * 255


def _side_brow_top_1d(
    shape: tuple[int, int],
    brow_pts: np.ndarray,
    io: float,
    width_ratio: float,
    y_shift_ratio: float,
    pad_up: float,
) -> np.ndarray:
    """单侧眉弧上缘；仅在眉 x 范围内有效，不外 extrapolate 到对侧。"""
    h, w = shape
    xx = np.arange(w, dtype=np.float32)
    y_out = np.full(w, float(h), dtype=np.float32)
    if len(brow_pts) < 2:
        return y_out
    y_c, x_l, x_r = _smooth_brow_arc_curve(shape, brow_pts, io, width_ratio, y_shift_ratio)
    active = (xx >= x_l) & (xx <= x_r)
    if np.any(active):
        y_out[active] = (y_c - pad_up - io * 0.003)[active]
    return y_out


def _forehead_brow_cutoff_y_map(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    io: float,
    y_shift_ratio: float,
    height_ratio: float,
    dilate_px: int,
    shrink_px: int,
) -> np.ndarray:
    """左右眉分别拟合，按列取较低上缘，避免侧脸时斜切额头。"""
    h, w = shape
    pad_up, _ = _brow_protect_thickness(io, dilate_px, shrink_px, height_ratio)
    y_cut = np.full(w, float(h), dtype=np.float32)
    for brow_ids, eye_ids in ((LEFT_BROW, LEFT_EYE), (RIGHT_BROW, RIGHT_EYE)):
        brow = _filter_side_brow_pts(pts(landmarks, brow_ids), pts(landmarks, eye_ids), io)
        top = _side_brow_top_1d(shape, brow, io, 1.0, y_shift_ratio, pad_up)
        y_cut = np.minimum(y_cut, top)
    return np.broadcast_to(y_cut, (h, w))


def _compute_forehead_lift_px(
    landmarks: np.ndarray,
    io: float,
    oval_pts: np.ndarray,
    fh_expand: float,
) -> float:
    brow_pts = np.vstack([pts(landmarks, LEFT_BROW), pts(landmarks, RIGHT_BROW)])
    if len(brow_pts) < 2 or len(oval_pts) < 3:
        return io * 0.20
    brow_bottom_y = float(np.percentile(brow_pts[:, 1], 72))
    span = max(brow_bottom_y - float(oval_pts[:, 1].min()), io * 0.20, 4.0)
    if fh_expand <= 1.0:
        return io * 0.20 * max(0.35, fh_expand)
    return io * 0.20 + (fh_expand - 1.0) * span


def _lift_face_oval_pts(
    landmarks: np.ndarray,
    io: float,
    forehead_height_ratio: float,
) -> np.ndarray:
    """沿脸廓弧线上抬至发际附近，供 mask 与绿色可视化轮廓共用。"""
    oval_pts = pts(landmarks, FACE_OVAL)
    if len(oval_pts) < 3:
        return oval_pts
    brow_pts = np.vstack([pts(landmarks, LEFT_BROW), pts(landmarks, RIGHT_BROW)])
    brow_bottom_y = (
        float(np.percentile(brow_pts[:, 1], 72))
        if len(brow_pts) >= 2
        else float(oval_pts[:, 1].min()) + io * 0.35
    )
    lift_px = _compute_forehead_lift_px(landmarks, io, oval_pts, forehead_height_ratio)
    if lift_px <= 0.5:
        return oval_pts
    lifted = oval_pts.copy()
    upper = lifted[:, 1] < brow_bottom_y - 1.0
    lifted[upper, 1] = np.maximum(0.0, lifted[upper, 1] - float(lift_px))
    return lifted


def _forehead_support_oval(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    face_mask_raw: np.ndarray,
    lift_px: float,
    brow_bottom_y: float,
) -> np.ndarray:
    """ratio>1 时把脸廓上缘弧线整体上抬，仍是一个连贯 oval，不做 OR 拼块。"""
    if lift_px <= 0.5:
        return face_mask_raw
    oval_pts = pts(landmarks, FACE_OVAL)
    if len(oval_pts) < 3:
        return face_mask_raw
    lifted = oval_pts.copy()
    upper = lifted[:, 1] < brow_bottom_y - 1.0
    lifted[upper, 1] = np.maximum(0.0, lifted[upper, 1] - float(lift_px))
    out = fill_poly_mask(shape, lifted)
    if out.sum() == 0:
        out = fill_hull_mask(shape, lifted)
    return out if out.sum() else face_mask_raw


def _build_lifted_face_mask_raw(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    io: float,
    params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    fh_expand = float(params.get("forehead_height_ratio", 1.28))
    contour_pts = _lift_face_oval_pts(landmarks, io, fh_expand)
    face_mask_raw = fill_poly_mask(shape, contour_pts)
    if face_mask_raw.sum() == 0:
        face_mask_raw = fill_hull_mask(shape, contour_pts)
    if face_mask_raw.sum() == 0:
        contour_pts = pts(landmarks, FACE_OVAL)
        face_mask_raw = fill_poly_mask(shape, contour_pts)
        if face_mask_raw.sum() == 0:
            face_mask_raw = fill_hull_mask(shape, contour_pts)
    return face_mask_raw, contour_pts


def _build_forehead_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    geo,
    io: float,
    face_mask_raw: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """额头 = 脸廓弧线 ∩ 眉线以上；ratio>1 只把同一条弧线整体上抬。"""
    del geo
    fh_expand = float(params.get("forehead_height_ratio", 1.0))
    brow_pts = np.vstack([pts(landmarks, LEFT_BROW), pts(landmarks, RIGHT_BROW)])
    if len(brow_pts) < 2:
        return np.zeros(shape, np.uint8)

    brow_bottom_y = float(np.percentile(brow_pts[:, 1], 72))
    lift_px = _compute_forehead_lift_px(landmarks, io, pts(landmarks, FACE_OVAL), fh_expand)
    oval = _forehead_support_oval(shape, landmarks, face_mask_raw, lift_px, brow_bottom_y)

    h, w = shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    brow_shrink = int(params.get("brow_protect_shrink_radius", 2))
    protect_radius = int(params.get("protect_expand_radius", 6))
    y_shift = float(params.get("brow_vertical_shift_ratio", 0.0))
    height_ratio = float(params.get("brow_protect_height_ratio", 1.0))
    y_cut = _forehead_brow_cutoff_y_map(
        shape, landmarks, io, y_shift, height_ratio, protect_radius, brow_shrink,
    )
    forehead = ((oval > 0) & (yy < y_cut)).astype(np.uint8) * 255

    if fh_expand < 1.0 and int((forehead > 0).sum()) > 0:
        ys, _xs = np.where(forehead > 0)
        bottom_y = int(ys.max())
        span = max(4, bottom_y - int(ys.min()))
        keep_from = bottom_y - int(span * max(0.35, fh_expand))
        forehead[yy < keep_from] = 0

    return forehead


def _clip_forehead(mask: np.ndarray, protect: np.ndarray) -> np.ndarray:
    """额头只排除五官保护区，形状仍沿脸廓弧线。"""
    out = (mask_to_uint8(mask) > 0) & (mask_to_uint8(protect) == 0)
    return out.astype(np.uint8) * 255


def _build_face_regions_simple(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    params: Dict[str, Any],
) -> RegionMasks:
    """简化模式：只保护眉/眼/嘴，其余皮肤统一作为可去高光区域。"""
    h, w = image_bgr.shape[:2]
    shape = (h, w)
    geo = get_geometry(landmarks, image_bgr.shape)
    io = max(1.0, geo.interocular)
    warnings: List[str] = []
    head_yaw = estimate_head_yaw_ratio(landmarks)

    face_mask_raw, face_contour_pts = _build_lifted_face_mask_raw(shape, landmarks, io, params)
    shrink_px = max(0, int(round(io * float(params.get("face_contour_shrink_ratio", 0.02)))))
    face_mask = morph(face_mask_raw, "erode", shrink_px) if shrink_px > 0 else face_mask_raw.copy()

    protect_radius = int(params.get("protect_expand_radius", 6))
    eye_extra = float(params.get("eye_protect_extra_radius", 0.5))
    brow_shrink = int(params.get("brow_protect_shrink_radius", 2))
    brow_height_ratio = float(params.get("brow_protect_height_ratio", 1.0))
    brow_width_ratio = float(params.get("brow_protect_width_ratio", 1.0))
    brow_y_shift = float(params.get("brow_vertical_shift_ratio", 0.12))
    lip_shrink = int(params.get("lip_protect_shrink_radius", 2))
    eye_radius = max(1, protect_radius + int(round(eye_extra)))
    lip_radius = max(1, protect_radius - lip_shrink)

    left_eye = _protect_feature(shape, landmarks, LEFT_EYE, eye_radius)
    right_eye = _protect_feature(shape, landmarks, RIGHT_EYE, eye_radius)
    lips = _protect_feature(shape, landmarks, OUTER_LIPS + INNER_LIPS, lip_radius)
    left_brow_protect = _build_brow_protect_mask(
        shape, landmarks, LEFT_BROW, LEFT_EYE, io, protect_radius, brow_shrink,
        brow_height_ratio, brow_width_ratio, brow_y_shift,
    )
    right_brow_protect = _build_brow_protect_mask(
        shape, landmarks, RIGHT_BROW, RIGHT_EYE, io, protect_radius, brow_shrink,
        brow_height_ratio, brow_width_ratio, brow_y_shift,
    )
    protect = cv2.bitwise_or(left_eye, right_eye)
    protect = cv2.bitwise_or(protect, lips)
    protect = cv2.bitwise_or(protect, left_brow_protect)
    protect = cv2.bitwise_or(protect, right_brow_protect)
    protect = cv2.bitwise_and(protect, face_mask_raw)

    skin = estimate_skin_mask(
        image_bgr,
        face_mask,
        protect,
        {"skin_mask_smooth_radius": params.get("skin_mask_smooth_radius", 3)},
    )
    treatable = (
        (mask_to_uint8(face_mask) > 0)
        & (mask_to_uint8(skin) > 0)
        & (mask_to_uint8(protect) == 0)
    ).astype(np.uint8) * 255

    empty = np.zeros(shape, np.uint8)
    if skin.sum() == 0:
        warnings.append("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点")
    if treatable.sum() == 0:
        warnings.append("可去高光皮肤区域为空，请检查人脸关键点")
    if abs(head_yaw) > 0.18:
        warnings.append(
            f"侧脸角度较大（yaw≈{head_yaw:.2f}），建议正脸或开启「人脸轻量对齐」"
        )

    masks = {
        "face_mask_raw": face_mask_raw,
        "face_mask": face_mask,
        "skin": skin,
        "protect": protect,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_brow_protect": left_brow_protect,
        "right_brow_protect": right_brow_protect,
        "left_brow_ridge": empty,
        "right_brow_ridge": empty,
        "lips": lips,
        "mouth": lips,
        "nose_bridge": empty,
        "nose_tip": empty,
        "forehead": empty,
        "left_cheek": empty,
        "right_cheek": empty,
        "chin": empty,
        "philtrum": empty,
        "highlight_candidate": treatable,
        "treatable_skin": treatable,
        "face_contour_pts": face_contour_pts,
    }
    return RegionMasks(masks=masks, geometry=geo, warnings=warnings)


def _use_simple_protect_mode(params: Dict[str, Any]) -> bool:
    if bool(params.get("key_region_detection_mode", False)):
        return False
    return bool(params.get("simple_protect_mode", True))


def build_face_regions(image_bgr: np.ndarray, landmarks: np.ndarray, params: Dict[str, Any]) -> RegionMasks:
    if _use_simple_protect_mode(params):
        return _build_face_regions_simple(image_bgr, landmarks, params)
    h, w = image_bgr.shape[:2]
    shape = (h, w)
    geo = get_geometry(landmarks, image_bgr.shape)
    io = max(1.0, geo.interocular)
    warnings: List[str] = []

    face_mask_raw, face_contour_pts = _build_lifted_face_mask_raw(shape, landmarks, io, params)
    shrink_px = max(0, int(round(io * float(params.get("face_contour_shrink_ratio", 0.02)))))
    face_mask = morph(face_mask_raw, "erode", shrink_px) if shrink_px > 0 else face_mask_raw.copy()

    protect_radius = int(params.get("protect_expand_radius", 6))
    eye_extra = float(params.get("eye_protect_extra_radius", 0.5))
    brow_shrink = int(params.get("brow_protect_shrink_radius", 2))
    brow_height_ratio = float(params.get("brow_protect_height_ratio", 1.0))
    brow_width_ratio = float(params.get("brow_protect_width_ratio", 1.0))
    brow_y_shift = float(params.get("brow_vertical_shift_ratio", 0.12))
    head_yaw = estimate_head_yaw_ratio(landmarks)
    lip_shrink = int(params.get("lip_protect_shrink_radius", 2))
    eye_radius = max(1, protect_radius + int(round(eye_extra)))
    lip_radius = max(1, protect_radius - lip_shrink)
    left_eye = _protect_feature(shape, landmarks, LEFT_EYE, eye_radius)
    right_eye = _protect_feature(shape, landmarks, RIGHT_EYE, eye_radius)
    left_brow_protect = _build_brow_protect_mask(
        shape, landmarks, LEFT_BROW, LEFT_EYE, io, protect_radius, brow_shrink,
        brow_height_ratio, brow_width_ratio, brow_y_shift,
    )
    right_brow_protect = _build_brow_protect_mask(
        shape, landmarks, RIGHT_BROW, RIGHT_EYE, io, protect_radius, brow_shrink,
        brow_height_ratio, brow_width_ratio, brow_y_shift,
    )
    lips = _protect_feature(shape, landmarks, OUTER_LIPS + INNER_LIPS, lip_radius)
    brow_union = cv2.bitwise_or(left_brow_protect, right_brow_protect)
    protect_brows = bool(params.get("protect_brows", False))
    protect = cv2.bitwise_or(left_eye, right_eye)
    protect = cv2.bitwise_or(protect, lips)
    if protect_brows:
        protect = cv2.bitwise_or(protect, brow_union)
    else:
        protect = cv2.bitwise_and(protect, cv2.bitwise_not(brow_union))
    if bool(params.get("philtrum_unprotect", True)):
        philtrum_w = float(params.get("philtrum_width_ratio", 0.16))
        protect = _carve_philtrum_from_protect(protect, landmarks, io, philtrum_w)
    protect = cv2.bitwise_and(protect, face_mask_raw)

    skin_params = {
        "skin_mask_smooth_radius": params.get("skin_mask_smooth_radius", 3)
    }
    skin = estimate_skin_mask(image_bgr, face_mask, protect, skin_params)

    # 鼻梁：由鼻根-鼻梁-鼻尖关键点连线加粗，不使用图像中心线。
    nose_bridge_expand = float(params.get("nose_bridge_expand_ratio", 0.12))
    bridge_thickness = max(2, int(round(io * nose_bridge_expand)))
    nose_bridge = line_mask(shape, pts(landmarks, NOSE_BRIDGE), bridge_thickness)
    nose_bridge = morph(nose_bridge, "dilate", max(1, int(round(io * 0.015))))

    # 鼻尖：由鼻翼宽度与鼻尖关键点构成椭圆，不使用固定中心。
    nose_pts = pts(landmarks, NOSE_WING)
    if len(nose_pts) >= 2:
        nose_width = float(np.max(nose_pts[:, 0]) - np.min(nose_pts[:, 0]))
        nose_height = float(np.max(nose_pts[:, 1]) - np.min(nose_pts[:, 1]))
    else:
        nose_width = io * 0.45
        nose_height = io * 0.35
    tip_expand = float(params.get("nose_tip_expand_ratio", 0.72))
    tip_center = np.asarray(landmarks[4, :2], dtype=np.float32) if len(landmarks) > 4 else np.asarray(geo.nose_tip, dtype=np.float32)
    nose_tip = ellipse_mask(shape, tip_center, (max(2, nose_width * tip_expand * 0.55), max(2, nose_height * tip_expand * 0.75)), geo.angle_deg)

    # 额头：上沿可超出 face oval / 肤色估计，仅排除保护区。
    forehead = _build_forehead_mask(shape, landmarks, geo, io, face_mask_raw, params)
    forehead = _clip_forehead(forehead, protect)
    philtrum = _build_philtrum_mask(shape, landmarks, io, params)

    # 姿态自适应局部坐标：由眼中心线和下巴方向确定，不依赖图片中心。
    origin = np.asarray(geo.nose_tip, dtype=np.float32)
    u_grid, v_grid = local_coordinate_grid(image_bgr.shape, origin, geo.x_axis, geo.y_axis)
    oval_p = pts(landmarks, FACE_OVAL)
    oval_u, oval_v = project_points(oval_p, origin, geo.x_axis, geo.y_axis)
    face_w_local = max(1.0, float(oval_u.max() - oval_u.min())) if len(oval_u) else io * 2.0
    face_h_local = max(1.0, float(oval_v.max() - oval_v.min())) if len(oval_v) else io * 2.8

    lower_eye_p = np.vstack([pts(landmarks, LEFT_LOWER_EYE), pts(landmarks, RIGHT_LOWER_EYE)])
    _, lower_eye_v = project_points(lower_eye_p, origin, geo.x_axis, geo.y_axis)
    mouth_p = pts(landmarks, OUTER_LIPS + MOUTH_CORNERS)
    _, mouth_v = project_points(mouth_p, origin, geo.x_axis, geo.y_axis)
    nosewing_p = pts(landmarks, NOSE_WING)
    nose_u, _ = project_points(nosewing_p, origin, geo.x_axis, geo.y_axis)
    chin_p = pts(landmarks, CHIN_AREA + [152])
    _, chin_v = project_points(chin_p, origin, geo.x_axis, geo.y_axis)

    cheek_upper = float(np.median(lower_eye_v)) - 0.05 * face_h_local if len(lower_eye_v) else -0.15 * face_h_local
    cheek_lower = float(np.percentile(mouth_v, 75)) + 0.07 * face_h_local if len(mouth_v) else 0.45 * face_h_local
    nose_half = max(io * 0.08, (float(np.percentile(nose_u, 85) - np.percentile(nose_u, 15)) / 2.0) if len(nose_u) else io * 0.12)
    cheek_expand = float(params.get("cheek_expand_ratio", 1.0))
    # cheek_expand > 1 会轻微放宽上下界，仍然基于局部关键点坐标。
    cheek_margin = (cheek_expand - 1.0) * 0.06 * face_h_local
    cheek_band = (v_grid >= cheek_upper - cheek_margin) & (v_grid <= cheek_lower + cheek_margin)
    midline_u = 0.0
    left_cheek = ((u_grid < midline_u - nose_half * 0.75) & cheek_band & (face_mask > 0)).astype(np.uint8) * 255
    right_cheek = ((u_grid > midline_u + nose_half * 0.75) & cheek_band & (face_mask > 0)).astype(np.uint8) * 255

    lower_lip_v = float(np.percentile(mouth_v, 90)) if len(mouth_v) else 0.35 * face_h_local
    chin_upper = lower_lip_v - 0.01 * face_h_local
    chin_lower = float(np.max(chin_v)) + 0.02 * face_h_local if len(chin_v) else 0.70 * face_h_local
    chin = ((v_grid >= chin_upper) & (v_grid <= chin_lower) & (face_mask > 0)).astype(np.uint8) * 255
    chin_expand = float(params.get("chin_expand_ratio", 1.0))
    if chin_expand > 1.01:
        chin = morph(chin, "dilate", int(round(io * 0.04 * (chin_expand - 1.0))))

    # 将候选区限制到真实皮肤与非五官保护区。
    nose_bridge = _clip_to_skin(nose_bridge, skin, protect)
    nose_tip = _clip_to_skin(nose_tip, skin, protect)
    chin = _clip_to_skin(chin, skin, protect)
    left_cheek = _clip_to_skin(left_cheek, skin, protect)
    right_cheek = _clip_to_skin(right_cheek, skin, protect)
    philtrum = _clip_forehead(philtrum, protect)

    # 眉弓 = 剩余集合：优先级 额/颊/眉/眼 > 眉弓（眼、眉重叠处删掉眉弓）
    priority = cv2.bitwise_or(forehead, cv2.bitwise_or(left_cheek, right_cheek))
    priority = cv2.bitwise_or(priority, cv2.bitwise_or(left_eye, right_eye))
    priority = cv2.bitwise_or(priority, cv2.bitwise_or(left_brow_protect, right_brow_protect))
    priority = cv2.bitwise_or(priority, cv2.bitwise_or(nose_bridge, nose_tip))
    if abs(head_yaw) > 0.12:
        eye_mid_x = float(geo.eye_mid[0])
        xx_side = np.arange(w, dtype=np.float32)[None, :]
        left_side = xx_side < (eye_mid_x - io * 0.04)
        right_side = xx_side > (eye_mid_x + io * 0.04)
    else:
        left_side = u_grid < midline_u - nose_half * 0.35
        right_side = u_grid > midline_u + nose_half * 0.35
    left_brow_ridge = _build_residual_brow_ridge_side(
        shape, pts(landmarks, LEFT_BROW), pts(landmarks, LEFT_EYE), io,
        face_mask, skin, priority, left_side,
        brow_height_ratio, brow_width_ratio, protect_radius, brow_shrink, brow_y_shift,
    )
    right_brow_ridge = _build_residual_brow_ridge_side(
        shape, pts(landmarks, RIGHT_BROW), pts(landmarks, RIGHT_EYE), io,
        face_mask, skin, priority, right_side,
        brow_height_ratio, brow_width_ratio, protect_radius, brow_shrink, brow_y_shift,
    )
    brow_ridge = cv2.bitwise_or(left_brow_ridge, right_brow_ridge)
    protect = cv2.bitwise_and(protect, cv2.bitwise_not(brow_ridge))

    forehead = cv2.bitwise_and(forehead, cv2.bitwise_not(brow_ridge))

    # 皮肤 mask 必须覆盖整个额头候选区，否则额头上沿会被当成非皮肤而跳过去高光。
    skin = cv2.bitwise_or(skin, forehead)

    # 区域之间互斥清理：脸颊不吞鼻子/嘴唇/额头/下巴/人中/眉弓。
    non_cheek = cv2.bitwise_or(cv2.bitwise_or(nose_bridge, nose_tip), cv2.bitwise_or(forehead, chin))
    non_cheek = cv2.bitwise_or(non_cheek, philtrum)
    non_cheek = cv2.bitwise_or(non_cheek, brow_ridge)
    non_cheek = cv2.bitwise_or(non_cheek, protect)
    left_cheek[non_cheek > 0] = 0
    right_cheek[non_cheek > 0] = 0

    candidate = np.zeros(shape, np.uint8)
    region_parts = [
        nose_bridge, nose_tip, forehead, left_cheek, right_cheek, chin, philtrum,
        left_brow_ridge, right_brow_ridge,
    ]
    for m in region_parts:
        candidate = cv2.bitwise_or(candidate, m)

    if skin.sum() == 0:
        warnings.append("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点")
    if candidate.sum() == 0:
        warnings.append("高光候选区域为空，请检查区域扩张参数")
    if abs(head_yaw) > 0.18:
        warnings.append(
            f"侧脸角度较大（yaw≈{head_yaw:.2f}），分区可能不准；建议使用正脸照片，或开启「人脸轻量对齐」"
        )
    elif abs(head_yaw) > 0.12:
        warnings.append(f"检测到轻微侧脸（yaw≈{head_yaw:.2f}），眉/额分区已按左右分别拟合")

    masks = {
        "face_mask_raw": face_mask_raw,
        "face_mask": face_mask,
        "skin": skin,
        "protect": protect,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_brow_protect": left_brow_protect,
        "right_brow_protect": right_brow_protect,
        "left_brow_ridge": left_brow_ridge,
        "right_brow_ridge": right_brow_ridge,
        "lips": lips,
        "mouth": lips,
        "nose_bridge": nose_bridge,
        "nose_tip": nose_tip,
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "chin": chin,
        "philtrum": philtrum,
        "highlight_candidate": candidate,
        "face_contour_pts": face_contour_pts,
    }
    return RegionMasks(masks=masks, geometry=geo, warnings=warnings)
