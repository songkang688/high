"""ONNX 版人脸关键点检测参考实现（纯 Python + ONNX Runtime + OpenCV）。

严格按 MediaPipe FaceLandmarker 图的 CPU 路径复刻前后处理：
  1. ImageToTensor（OpenCV warpPerspective 路径）：
     - 检测器：整图 letterbox 到 128x128，归一化 [-1,1]，零填充；
     - 关键点：按旋转矩形裁剪到 256x256，归一化 [0,1]。
  2. BlazeFace SSD anchors：4 层 stride [8,16,16,16]，共 896 个，fixed_anchor_size。
  3. TensorsToDetections：reverse_output_order，x/y/w/h_scale=128，sigmoid 分数裁剪 ±100。
  4. 加权 NMS（IoU 阈值 0.3，WEIGHTED 模式）。
  5. DetectionsToRects：双眼关键点计算旋转角（目标 0°）。
  6. RectTransformation：scale 1.5、square_long。
  7. LandmarkProjection：把 256 空间归一化坐标投影回原图。

该实现与 C++ 版 face_landmarker_ort.cpp 一一对应；tools/validate_onnx_landmarker.py
用它和 MediaPipe 官方实现对拍。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

DETECTOR_INPUT = 128
LANDMARK_INPUT = 256
NUM_LANDMARKS = 478
MIN_SUPPRESSION_THRESHOLD = 0.3
RECT_SCALE = 1.5


@dataclass
class NormRect:
    x_center: float  # 归一化
    y_center: float
    width: float
    height: float
    rotation: float  # 弧度


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # 归一化 xmin, ymin, w, h
    keypoints: np.ndarray  # (6, 2) 归一化
    score: float


def build_blazeface_anchors() -> np.ndarray:
    """ssd_anchors_calculator.cc 的固定参数版本：返回 (896, 2) 的 anchor 中心。

    参数：num_layers=4, strides=[8,16,16,16], min_scale=0.1484375, max_scale=0.75,
    aspect_ratios=[1.0], interpolated_scale_aspect_ratio=1.0, anchor_offset=0.5,
    fixed_anchor_size=true（w=h=1，因此只需中心坐标）。
    """
    strides = [8, 16, 16, 16]
    anchors = []
    layer_id = 0
    while layer_id < len(strides):
        last_same_stride_layer = layer_id
        repeats = 0
        while (
            last_same_stride_layer < len(strides)
            and strides[last_same_stride_layer] == strides[layer_id]
        ):
            last_same_stride_layer += 1
            repeats += 2  # aspect_ratio 1.0 + interpolated 1.0 → 每层 2 个
        stride = strides[layer_id]
        feature_map_size = DETECTOR_INPUT // stride
        for y in range(feature_map_size):
            y_center = (y + 0.5) * stride / DETECTOR_INPUT
            for x in range(feature_map_size):
                x_center = (x + 0.5) * stride / DETECTOR_INPUT
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.asarray(anchors, dtype=np.float32)


def _image_to_tensor(
    image_bgr: np.ndarray,
    roi: NormRect,
    out_size: int,
    out_min: float,
    out_max: float,
) -> np.ndarray:
    """image_to_tensor_converter_opencv.cc 的等价实现。返回 (1,H,W,3) RGB float32。"""
    h, w = image_bgr.shape[:2]
    center = (roi.x_center * w, roi.y_center * h)
    size = (roi.width * w, roi.height * h)
    rotated = (center, size, math.degrees(roi.rotation))
    src_points = cv2.boxPoints(rotated).astype(np.float32)
    dst_points = np.array(
        [[0, out_size], [0, 0], [out_size, 0], [out_size, out_size]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(
        image_bgr, matrix, (out_size, out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    scale = (out_max - out_min) / 255.0
    tensor = rgb.astype(np.float32) * scale + out_min
    return tensor[None, ...]


def _pad_roi_to_square(roi: NormRect, image_w: int, image_h: int) -> NormRect:
    """keep_aspect_ratio=true 时把 ROI 补成输出宽高比（此处输出为正方形）。"""
    abs_w = roi.width * image_w
    abs_h = roi.height * image_h
    if abs_h / abs_w > 1.0:
        abs_w = abs_h
    else:
        abs_h = abs_w
    return NormRect(roi.x_center, roi.y_center, abs_w / image_w, abs_h / image_h, roi.rotation)


def _decode_detections(
    raw_boxes: np.ndarray, raw_scores: np.ndarray, anchors: np.ndarray, min_score: float
) -> List[Detection]:
    """tensors_to_detections_calculator.cc：num_coords=16、reverse_output_order=true。"""
    logits = np.clip(raw_scores.reshape(-1).astype(np.float64), -100.0, 100.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    keep = np.where(scores >= min_score)[0]
    out: List[Detection] = []
    for i in keep:
        r = raw_boxes[i]
        ax, ay = float(anchors[i, 0]), float(anchors[i, 1])
        xc = r[0] / DETECTOR_INPUT + ax
        yc = r[1] / DETECTOR_INPUT + ay
        bw = r[2] / DETECTOR_INPUT
        bh = r[3] / DETECTOR_INPUT
        kps = np.empty((6, 2), dtype=np.float64)
        for k in range(6):
            kps[k, 0] = r[4 + 2 * k] / DETECTOR_INPUT + ax
            kps[k, 1] = r[4 + 2 * k + 1] / DETECTOR_INPUT + ay
        out.append(
            Detection(
                bbox=(float(xc - bw / 2), float(yc - bh / 2), float(bw), float(bh)),
                keypoints=kps,
                score=float(scores[i]),
            )
        )
    return out


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    denom = aw * ah + bw * bh - inter
    return inter / denom if denom > 0 else 0.0


def _weighted_nms(detections: List[Detection]) -> List[Detection]:
    """non_max_suppression_calculator.cc 的 WEIGHTED 算法。"""
    remained = sorted(detections, key=lambda d: d.score, reverse=True)
    out: List[Detection] = []
    while remained:
        top = remained[0]
        candidates = [d for d in remained if _iou(d.bbox, top.bbox) > MIN_SUPPRESSION_THRESHOLD]
        remained = [d for d in remained if _iou(d.bbox, top.bbox) <= MIN_SUPPRESSION_THRESHOLD]
        if candidates:
            total = sum(d.score for d in candidates)
            x0 = sum((d.bbox[0]) * d.score for d in candidates) / total
            y0 = sum((d.bbox[1]) * d.score for d in candidates) / total
            x1 = sum((d.bbox[0] + d.bbox[2]) * d.score for d in candidates) / total
            y1 = sum((d.bbox[1] + d.bbox[3]) * d.score for d in candidates) / total
            kps = sum((d.keypoints * d.score for d in candidates), np.zeros((6, 2))) / total
            out.append(Detection((x0, y0, x1 - x0, y1 - y0), kps, top.score))
        else:
            out.append(top)
    return out


def _normalize_radians(angle: float) -> float:
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))


def _detection_to_rect(det: Detection, image_w: int, image_h: int) -> NormRect:
    """detections_to_rects_calculator.cc：旋转向量 = 关键点 0(左眼)→1(右眼)，目标 0°。"""
    x0 = det.keypoints[0, 0] * image_w
    y0 = det.keypoints[0, 1] * image_h
    x1 = det.keypoints[1, 0] * image_w
    y1 = det.keypoints[1, 1] * image_h
    rotation = _normalize_radians(-math.atan2(-(y1 - y0), x1 - x0))
    bx, by, bw, bh = det.bbox
    return NormRect(bx + bw / 2, by + bh / 2, bw, bh, rotation)


def _transform_rect(rect: NormRect, image_w: int, image_h: int) -> NormRect:
    """rect_transformation_calculator.cc：scale_x=scale_y=1.5, square_long=true。"""
    w = rect.width * image_w * RECT_SCALE
    h = rect.height * image_h * RECT_SCALE
    long_side = max(w, h)
    return NormRect(rect.x_center, rect.y_center, long_side / image_w, long_side / image_h, rect.rotation)


def _project_landmarks(raw: np.ndarray, rect: NormRect) -> np.ndarray:
    """landmark_projection_calculator.cc：把 [0,1] 张量坐标投影到归一化图像坐标。"""
    pts = raw.reshape(NUM_LANDMARKS, 3).astype(np.float64) / LANDMARK_INPUT
    x = pts[:, 0] - 0.5
    y = pts[:, 1] - 0.5
    c, s = math.cos(rect.rotation), math.sin(rect.rotation)
    nx = (x * c - y * s) * rect.width + rect.x_center
    ny = (x * s + y * c) * rect.height + rect.y_center
    nz = pts[:, 2] * rect.width
    return np.stack([nx, ny, nz], axis=1)


class OnnxFaceLandmarker:
    """与 MediaPipe FaceLandmarker(IMAGE 模式) 对应的 ONNX Runtime 实现。"""

    def __init__(self, model_dir: str | Path, num_faces: int = 4,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        self._det = ort.InferenceSession(
            str(model_dir / "face_detector.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._lmk = ort.InferenceSession(
            str(model_dir / "face_landmarks_detector.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._anchors = build_blazeface_anchors()
        self.num_faces = int(num_faces)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """返回 [(landmarks (478,3) 归一化坐标, presence 分数)]，顺序与检测分数一致。"""
        h, w = image_bgr.shape[:2]
        roi = NormRect(0.5, 0.5, 1.0, 1.0, 0.0)
        padded = _pad_roi_to_square(roi, w, h)
        tensor = _image_to_tensor(image_bgr, padded, DETECTOR_INPUT, -1.0, 1.0)
        raw_boxes, raw_scores = self._det.run(None, {self._det.get_inputs()[0].name: tensor})
        dets = _decode_detections(
            raw_boxes[0].astype(np.float64), raw_scores[0], self._anchors,
            self.min_detection_confidence,
        )
        # 检测坐标基于 letterbox 后的张量空间，投影回原图归一化坐标。
        pw = padded.width * w
        ph = padded.height * h
        projected: List[Detection] = []
        for d in dets:
            bx, by, bw, bh = d.bbox
            nx = (padded.x_center * w + (bx + bw / 2 - 0.5) * pw) / w
            ny = (padded.y_center * h + (by + bh / 2 - 0.5) * ph) / h
            nbw = bw * pw / w
            nbh = bh * ph / h
            kps = d.keypoints.copy()
            kps[:, 0] = (padded.x_center * w + (kps[:, 0] - 0.5) * pw) / w
            kps[:, 1] = (padded.y_center * h + (kps[:, 1] - 0.5) * ph) / h
            projected.append(
                Detection((nx - nbw / 2, ny - nbh / 2, nbw, nbh), kps, d.score)
            )
        merged = _weighted_nms(projected)[: self.num_faces]

        results: List[Tuple[np.ndarray, float]] = []
        for det in merged:
            rect = _transform_rect(_detection_to_rect(det, w, h), w, h)
            tensor = _image_to_tensor(image_bgr, rect, LANDMARK_INPUT, 0.0, 1.0)
            outs = self._lmk.run(None, {self._lmk.get_inputs()[0].name: tensor})
            raw_landmarks = outs[0].reshape(-1)
            # Identity_1 是 sigmoid 前的 face flag logit（老图里由 TensorsToFloats 加 SIGMOID）。
            presence = float(1.0 / (1.0 + math.exp(-float(outs[1].reshape(-1)[0]))))
            if presence < self.min_presence_confidence:
                continue
            results.append((_project_landmarks(raw_landmarks, rect), presence))
        return results

    def detect_pixels(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """与 highlight_removal.face_detect 一致的像素坐标格式：x*w, y*h, z*max(w,h)。"""
        h, w = image_bgr.shape[:2]
        out = []
        for lm, score in self.detect(image_bgr):
            px = lm.copy()
            px[:, 0] *= w
            px[:, 1] *= h
            px[:, 2] *= max(w, h)
            out.append((px.astype(np.float32), score))
        return out
