from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .face_detect import FaceResult, draw_faces
from .face_landmarks import draw_keypoints
from .face_regions import RegionMasks
from .utils import colorize_mask, mask_to_uint8, overlay_mask, put_text_cn


REGION_COLORS = {
    "protect": (0, 0, 255),
    "treatable_skin": (0, 220, 255),
    "nose_bridge": (255, 0, 0),
    "nose_tip": (255, 128, 0),
    "forehead": (0, 255, 255),
    "left_brow_ridge": (0, 220, 0),
    "right_brow_ridge": (0, 200, 0),
    "philtrum": (0, 200, 200),
    "left_cheek": (255, 0, 255),
    "right_cheek": (180, 0, 255),
    "chin": (0, 128, 255),
}

REGION_LABELS = {
    "protect": "五官保护",
    "treatable_skin": "可去高光皮肤",
    "nose_bridge": "鼻梁",
    "nose_tip": "鼻尖",
    "forehead": "额头",
    "left_brow_ridge": "眉弓处",
    "right_brow_ridge": "眉弓处",
    "philtrum": "人中",
    "left_cheek": "左脸颊",
    "right_cheek": "右脸颊",
    "chin": "下巴",
}

SIMPLE_PROTECT_PARTS = (
    ("left_brow_protect", "左眉"),
    ("right_brow_protect", "右眉"),
    ("left_eye", "左眼"),
    ("right_eye", "右眼"),
    ("lips", "嘴巴"),
)


def mask_preview(mask: np.ndarray) -> np.ndarray:
    return colorize_mask(mask)


def draw_landmark_view(image_bgr: np.ndarray, face: FaceResult, all_faces=None, show_index: bool = False) -> np.ndarray:
    out = image_bgr.copy()
    if all_faces is not None:
        out = draw_faces(out, all_faces, face)
    if face is not None and face.landmarks is not None:
        out = draw_keypoints(out, face.landmarks, show_index=show_index)
    return out


def _draw_face_contour(out: np.ndarray, regions: RegionMasks, face: Optional[FaceResult] = None) -> np.ndarray:
    contour = regions.masks.get("face_contour_pts")
    if contour is not None and len(contour) >= 3:
        h, w = out.shape[:2]
        pts_int = np.round(np.asarray(contour, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        pts_int[:, 0, 0] = np.clip(pts_int[:, 0, 0], 0, w - 1)
        pts_int[:, 0, 1] = np.clip(pts_int[:, 0, 1], 0, h - 1)
        cv2.polylines(out, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        return out
    if face is None:
        return out
    x, y, bw, bh = face.bbox
    fh = mask_to_uint8(regions.masks.get("forehead", np.zeros(out.shape[:2], np.uint8)))
    if int((fh > 0).sum()) > 0:
        y_top = int(np.where(fh > 0)[0].min())
        y = min(y, y_top)
        bh = max(bh, (face.bbox[1] + face.bbox[3]) - y)
    elif "treatable_skin" in regions.masks:
        brow_parts = [
            regions.masks.get("left_brow_protect"),
            regions.masks.get("right_brow_protect"),
        ]
        ys = []
        for part in brow_parts:
            m = mask_to_uint8(part) if part is not None else np.zeros(out.shape[:2], np.uint8)
            if int((m > 0).sum()) > 0:
                ys.append(int(np.where(m > 0)[0].min()))
        if ys:
            y = min(y, min(ys) - max(2, int(bw * 0.02)))
            bh = max(bh, (face.bbox[1] + face.bbox[3]) - y)
    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    return out


def draw_region_view(image_bgr: np.ndarray, regions: RegionMasks, face: Optional[FaceResult] = None) -> np.ndarray:
    out = image_bgr.copy()

    if "treatable_skin" in regions.masks:
        treatable = mask_to_uint8(regions.masks["treatable_skin"])
        if int((treatable > 0).sum()) > 0:
            out = overlay_mask(out, treatable, REGION_COLORS["treatable_skin"], alpha=0.28)
            ys, xs = np.where(treatable > 0)
            out = put_text_cn(out, "可去高光皮肤", (int(xs.mean()), int(ys.mean())), REGION_COLORS["treatable_skin"], font_size=18)
        for key, label in SIMPLE_PROTECT_PARTS:
            if key not in regions.masks:
                continue
            part = mask_to_uint8(regions.masks[key])
            if int((part > 0).sum()) < 10:
                continue
            out = overlay_mask(out, part, REGION_COLORS["protect"], alpha=0.45)
            ys, xs = np.where(part > 0)
            out = put_text_cn(out, label, (int(xs.mean()), int(ys.mean())), REGION_COLORS["protect"], font_size=18)
        if face is not None:
            out = _draw_face_contour(out, regions, face)
        return out

    protect = mask_to_uint8(regions.masks.get("protect", np.zeros(image_bgr.shape[:2], np.uint8)))
    region_keys = [
        "forehead", "philtrum",
        "left_brow_ridge", "right_brow_ridge",
        "left_cheek", "right_cheek", "chin", "nose_bridge", "nose_tip",
    ]
    for key in region_keys:
        if key not in regions.masks:
            continue
        out = overlay_mask(out, regions.masks[key], REGION_COLORS.get(key, (255, 255, 255)), alpha=0.32)
    if protect.sum() > 0:
        out = overlay_mask(out, protect, REGION_COLORS["protect"], alpha=0.45)
    for key, label in REGION_LABELS.items():
        if key not in regions.masks and key != "protect":
            continue
        m = protect > 0 if key == "protect" else mask_to_uint8(regions.masks[key]) > 0
        if m.sum() < 10:
            continue
        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        color = REGION_COLORS["protect"] if key == "protect" else REGION_COLORS.get(key, (255, 255, 255))
        out = put_text_cn(out, label, (cx, cy), color, font_size=18)
    if face is not None:
        out = _draw_face_contour(out, regions, face)
    return out


def combine_masks(masks: Dict[str, np.ndarray], keys) -> np.ndarray:
    first = next(iter(masks.values()))
    out = np.zeros_like(first, dtype=np.uint8)
    for k in keys:
        if k in masks:
            out = cv2.bitwise_or(out, mask_to_uint8(masks[k]))
    return out


def draw_highlight_view(image_bgr: np.ndarray, highlight_mask: np.ndarray) -> np.ndarray:
    return overlay_mask(image_bgr, highlight_mask, (0, 0, 255), alpha=0.55)
