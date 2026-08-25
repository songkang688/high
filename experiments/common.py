"""检测审计与 A/B 实验共享工具。

仅供 experiments/ 下脚本使用，不改变 highlight_removal/ 生产行为。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy

bootstrap_before_numpy()

import cv2
import numpy as np

from highlight_removal.face_detect import FaceResult
from highlight_removal.face_regions import build_face_regions
from highlight_removal.utils import mask_to_uint8

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "experiments" / "out"
LM_DIR = OUT_DIR / "landmarks"
NAMES = [str(i) for i in range(1, 18)]


def imread(name: str) -> np.ndarray:
    img = cv2.imread(str(DATA_DIR / f"{name}.png"), cv2.IMREAD_COLOR)
    assert img is not None, name
    return img


def save_face(name: str, face: FaceResult) -> None:
    LM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LM_DIR / f"{name}.npz",
        landmarks=face.landmarks,
        bbox=np.asarray(face.bbox, dtype=np.int64),
        confidence=np.float64(face.confidence),
        detector=np.str_(face.detector),
        face_index=np.int64(face.face_index),
    )


def load_face(name: str) -> FaceResult:
    d = np.load(LM_DIR / f"{name}.npz")
    return FaceResult(
        bbox=tuple(int(v) for v in d["bbox"]),
        confidence=float(d["confidence"]),
        landmarks=d["landmarks"],
        detector=str(d["detector"]),
        face_index=int(d["face_index"]),
    )


def lap_var(gray: np.ndarray, mask: np.ndarray) -> float:
    m = mask_to_uint8(mask) > 0
    if int(m.sum()) < 16:
        return 0.0
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[m]))


def ring_of(mask: np.ndarray, allowed: np.ndarray, radius: int = 15) -> np.ndarray:
    """mask 周围一圈可比较的正常皮肤邻域。"""
    m = (mask_to_uint8(mask) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.dilate(m, k) > 0
    return ring & (~(m > 0)) & (mask_to_uint8(allowed) > 0)


def residual_shine(result_bgr: np.ndarray, eval_mask: np.ndarray, treatable: np.ndarray) -> Dict[str, float]:
    """处理后高光候选区内 mean L 相对邻域皮肤是否仍偏高。"""
    L = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    m = mask_to_uint8(eval_mask) > 0
    ring = ring_of(eval_mask, treatable)
    if int(m.sum()) < 16 or int(ring.sum()) < 16:
        return {"mean_l_in": 0.0, "mean_l_ring": 0.0, "residual_dl": 0.0}
    mi, mr = float(L[m].mean()), float(L[ring].mean())
    return {"mean_l_in": round(mi, 3), "mean_l_ring": round(mr, 3), "residual_dl": round(mi - mr, 3)}


def eval_partition_masks(image_bgr: np.ndarray, face: FaceResult, region_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """几何分区（仅评估用）：常用模式 simple 分区没有额头/鼻/颊，这里用
    key_region_detection_mode 的几何区域做粗分。"""
    p = dict(region_params)
    p["simple_protect_mode"] = False
    p["key_region_detection_mode"] = True
    regions = build_face_regions(image_bgr, face.landmarks, p)
    nose = cv2.bitwise_or(mask_to_uint8(regions.masks["nose_bridge"]), mask_to_uint8(regions.masks["nose_tip"]))
    cheeks = cv2.bitwise_or(mask_to_uint8(regions.masks["left_cheek"]), mask_to_uint8(regions.masks["right_cheek"]))
    return {
        "forehead": mask_to_uint8(regions.masks["forehead"]),
        "nose": nose,
        "cheeks": cheeks,
        "chin": mask_to_uint8(regions.masks["chin"]),
    }


def zone_report(image_bgr: np.ndarray, zone: np.ndarray, soft_mask: np.ndarray, bright_delta: float = 7.0) -> Dict[str, Any]:
    """区域是否有亮斑、亮斑被 mask 点亮的比例。"""
    L = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    z = mask_to_uint8(zone) > 0
    if int(z.sum()) < 32:
        return {"zone_px": int(z.sum()), "bright_px": 0, "lit": False, "bright_covered": 0.0}
    p50 = float(np.percentile(L[z], 50))
    bright = z & (L >= p50 + bright_delta)
    soft = mask_to_uint8(soft_mask) > 0
    lit = bool((soft & z).sum() > 0)
    cov = float((bright & soft).sum() / max(1, bright.sum()))
    return {
        "zone_px": int(z.sum()),
        "bright_px": int(bright.sum()),
        "lit": lit,
        "bright_covered": round(cov, 4),
    }


def component_count(mask: np.ndarray) -> int:
    m = (mask_to_uint8(mask) > 0).astype(np.uint8)
    num, _ = cv2.connectedComponents(m, 8)
    return int(num - 1)


def masked_ssim(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    m = mask_to_uint8(mask) > 0
    if int(m.sum()) < 64:
        return 1.0
    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    _, smap = structural_similarity(ga, gb, full=True, data_range=255)
    return float(smap[m].mean())


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")
