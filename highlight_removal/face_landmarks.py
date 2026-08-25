from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from .utils import put_text_cn

# MediaPipe FaceMesh 468/478 点常用拓扑索引。
# 这些索引是人脸语义点，不是图像固定坐标，也不隐含图像中心。
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_TIP = [1, 4, 5, 19, 94, 195, 197, 2]
NOSE_LEFT_SIDE = [49, 98, 97, 2, 94]
NOSE_RIGHT_SIDE = [279, 327, 326, 2, 94]
NOSE_WING = [49, 98, 97, 2, 326, 327, 279]
MOUTH_CORNERS = [61, 291]
CHIN_AREA = [152, 148, 176, 149, 150, 136, 172, 58, 172, 378, 400, 377, 152, 365, 379]
# 额头区域由上面部轮廓与眉毛构成；不是固定矩形。
FOREHEAD_TOP = [103, 67, 109, 10, 338, 297, 332]
FOREHEAD_SIDE = [54, 103, 67, 109, 10, 338, 297, 332, 284]
LEFT_LOWER_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_LOWER_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]


@dataclass
class FaceGeometry:
    left_eye_center: Tuple[float, float]
    right_eye_center: Tuple[float, float]
    eye_mid: Tuple[float, float]
    nose_tip: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]
    chin: Tuple[float, float]
    interocular: float
    face_bbox: Tuple[int, int, int, int]
    x_axis: np.ndarray
    y_axis: np.ndarray
    angle_deg: float


def valid_indices(landmarks: np.ndarray, indices: Iterable[int]) -> list[int]:
    n = len(landmarks)
    return [i for i in indices if 0 <= i < n and np.all(np.isfinite(landmarks[i, :2]))]


def pts(landmarks: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    ids = valid_indices(landmarks, indices)
    if not ids:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(landmarks[ids, :2], dtype=np.float32)


def estimate_head_yaw_ratio(landmarks: np.ndarray) -> float:
    """估计侧脸程度：0≈正脸，|值|越大越侧（相对两眼间距）。"""
    le = np.array(center_of(landmarks, LEFT_EYE), dtype=np.float32)
    re = np.array(center_of(landmarks, RIGHT_EYE), dtype=np.float32)
    nose = np.array(center_of(landmarks, [4]), dtype=np.float32)
    vec = re - le
    io = float(np.linalg.norm(vec))
    if io < 1.0:
        return 0.0
    x_axis = vec / io
    eye_mid = (le + re) * 0.5
    return float(np.dot(nose - eye_mid, x_axis) / io)


def _filter_side_brow_pts(brow_pts: np.ndarray, eye_pts: np.ndarray, io: float) -> np.ndarray:
    """侧脸时去掉偏离该侧眼眶的眉点，避免另一侧折叠点拉歪弧线。"""
    if len(brow_pts) < 2 or len(eye_pts) < 2:
        return brow_pts
    ex0, ex1 = float(eye_pts[:, 0].min()), float(eye_pts[:, 0].max())
    ey_top = float(eye_pts[:, 1].min())
    keep = (
        (brow_pts[:, 0] >= ex0 - io * 0.14)
        & (brow_pts[:, 0] <= ex1 + io * 0.14)
        & (brow_pts[:, 1] <= ey_top + io * 0.06)
    )
    filtered = brow_pts[keep]
    if len(filtered) >= 2 and float(filtered[:, 0].max() - filtered[:, 0].min()) >= io * 0.05:
        return filtered
    return brow_pts


def center_of(landmarks: np.ndarray, indices: Iterable[int]) -> Tuple[float, float]:
    p = pts(landmarks, indices)
    if len(p) == 0:
        return (0.0, 0.0)
    c = p.mean(axis=0)
    return (float(c[0]), float(c[1]))


def get_geometry(landmarks: np.ndarray, image_shape) -> FaceGeometry:
    le = np.array(center_of(landmarks, LEFT_EYE), dtype=np.float32)
    re = np.array(center_of(landmarks, RIGHT_EYE), dtype=np.float32)
    eye_mid = (le + re) / 2.0
    nose_tip = np.array(center_of(landmarks, [4]), dtype=np.float32)
    mouth_left = np.array(center_of(landmarks, [61]), dtype=np.float32)
    mouth_right = np.array(center_of(landmarks, [291]), dtype=np.float32)
    chin = np.array(center_of(landmarks, [152]), dtype=np.float32)
    vec = re - le
    interocular = float(np.linalg.norm(vec))
    if interocular < 1:
        interocular = float(max(image_shape[:2]) * 0.1)
        vec = np.array([1.0, 0.0], dtype=np.float32)
    x_axis = vec / (np.linalg.norm(vec) + 1e-6)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
    if np.dot(chin - eye_mid, y_axis) < 0:
        y_axis = -y_axis
    angle = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
    bbox = bbox_from_landmarks(landmarks, image_shape)
    return FaceGeometry(
        left_eye_center=(float(le[0]), float(le[1])),
        right_eye_center=(float(re[0]), float(re[1])),
        eye_mid=(float(eye_mid[0]), float(eye_mid[1])),
        nose_tip=(float(nose_tip[0]), float(nose_tip[1])),
        mouth_left=(float(mouth_left[0]), float(mouth_left[1])),
        mouth_right=(float(mouth_right[0]), float(mouth_right[1])),
        chin=(float(chin[0]), float(chin[1])),
        interocular=interocular,
        face_bbox=bbox,
        x_axis=x_axis,
        y_axis=y_axis,
        angle_deg=angle,
    )


def bbox_from_landmarks(landmarks: np.ndarray, image_shape) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    p = pts(landmarks, FACE_OVAL)
    if len(p) == 0:
        p = landmarks[:, :2]
    x0, y0 = np.floor(p.min(axis=0)).astype(int)
    x1, y1 = np.ceil(p.max(axis=0)).astype(int)
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    return (x0, y0, max(0, x1 - x0 + 1), max(0, y1 - y0 + 1))


def project_points(points_xy: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points_xy, dtype=np.float32) - origin.reshape(1, 2)
    u = p @ x_axis.reshape(2, 1)
    v = p @ y_axis.reshape(2, 1)
    return u.reshape(-1), v.reshape(-1)


def local_coordinate_grid(image_shape, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - origin[0]
    dy = yy - origin[1]
    u = dx * x_axis[0] + dy * x_axis[1]
    v = dx * y_axis[0] + dy * y_axis[1]
    return u, v


def all_region_indices() -> Dict[str, list[int]]:
    return {
        "左眼": LEFT_EYE,
        "右眼": RIGHT_EYE,
        "左眉": LEFT_BROW,
        "右眉": RIGHT_BROW,
        "嘴唇": OUTER_LIPS + INNER_LIPS,
        "鼻梁": NOSE_BRIDGE,
        "鼻尖": NOSE_TIP,
        "面部轮廓": FACE_OVAL,
        "额头": FOREHEAD_SIDE + LEFT_BROW + RIGHT_BROW,
        "下巴": CHIN_AREA,
    }


def draw_keypoints(image_bgr: np.ndarray, landmarks: np.ndarray, show_index: bool = False) -> np.ndarray:
    out = image_bgr.copy()
    if landmarks is None or len(landmarks) == 0:
        return out
    for i, p in enumerate(landmarks[:, :2]):
        if not np.all(np.isfinite(p)):
            continue
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), 1, (0, 255, 255), -1, cv2.LINE_AA)
            if show_index and i % 5 == 0:
                cv2.putText(out, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)
    geo = get_geometry(landmarks, image_bgr.shape)
    for name, center, color in [
        ("左眼", geo.left_eye_center, (255, 0, 0)),
        ("右眼", geo.right_eye_center, (255, 0, 0)),
        ("鼻尖", geo.nose_tip, (0, 255, 0)),
        ("左嘴角", geo.mouth_left, (0, 0, 255)),
        ("右嘴角", geo.mouth_right, (0, 0, 255)),
    ]:
        cv2.circle(out, (int(center[0]), int(center[1])), 4, color, -1, cv2.LINE_AA)
        out = put_text_cn(out, name, (int(center[0]) + 4, int(center[1]) - 4), color, font_size=16)
    return out
