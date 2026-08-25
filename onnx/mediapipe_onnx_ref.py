# -*- coding: utf-8 -*-
"""MediaPipe FaceLandmarker 前后处理的 ONNX 参考实现（Python 原型）。

该文件与 C++ 版 `cpp/src/landmarker_onnx.cpp` 的算法逐行对应，用于：
1. 验证 anchor 解码 / 加权 NMS / ROI 旋转裁剪 / 关键点投影是否与
   MediaPipe Tasks 官方运行时输出一致；
2. 作为 C++ 移植的可执行规范。

流程（与 mediapipe/tasks face_landmarker_graph 一致）：
- 人脸检测：BlazeFace short-range，128x128 letterbox，[-1,1] 归一化，
  SSD anchors（4 层，strides 8/16/16/16，896 个），sigmoid 分数，
  加权 NMS（IoU 0.3）；
- ROI：由检测框 + 双眼关键点计算旋转角，RectTransformation
  scale 1.5 / square_long；
- 关键点：256x256 旋转裁剪（warpPerspective），[0,1] 归一化，
  输出 478×3，除以 256 归一化后经 LandmarkProjection 投影回原图。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# SSD anchors（对应 mediapipe SsdAnchorsCalculator，参数取自 face_detector_graph）
# ---------------------------------------------------------------------------

def build_blazeface_anchors() -> np.ndarray:
    """返回 (896, 4)：x_center, y_center, w, h（tensor 归一化坐标，w=h=1）。"""
    strides = [8, 16, 16, 16]
    anchors = []
    layer_id = 0
    while layer_id < len(strides):
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < len(strides)
               and strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios=[1.0] + interpolated_scale_aspect_ratio=1.0 → 每层 2 个
            repeats += 2
        stride = strides[layer_id]
        feature_h = 128 // stride
        feature_w = 128 // stride
        for y in range(feature_h):
            for x in range(feature_w):
                for _ in range(repeats):
                    anchors.append([(x + 0.5) / feature_w, (y + 0.5) / feature_h, 1.0, 1.0])
        layer_id = last_same_stride_layer
    return np.asarray(anchors, dtype=np.float32)


# ---------------------------------------------------------------------------
# 检测解码（对应 TensorsToDetectionsCalculator，x/y/w/h_scale=128）
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    score: float
    box: np.ndarray        # (4,) xmin, ymin, xmax, ymax（tensor 归一化坐标）
    keypoints: np.ndarray  # (6, 2)


def decode_detections(regressors: np.ndarray, scores_logit: np.ndarray,
                      anchors: np.ndarray, min_score: float) -> List[Detection]:
    raw = regressors.reshape(-1, 16).astype(np.float32)
    logits = np.clip(scores_logit.reshape(-1).astype(np.float32), -100.0, 100.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    out: List[Detection] = []
    for i in range(raw.shape[0]):
        if scores[i] < min_score:
            continue
        ax, ay, aw, ah = anchors[i]
        cx = raw[i, 0] / 128.0 * aw + ax
        cy = raw[i, 1] / 128.0 * ah + ay
        w = raw[i, 2] / 128.0 * aw
        h = raw[i, 3] / 128.0 * ah
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        kps = np.zeros((6, 2), dtype=np.float32)
        for k in range(6):
            kps[k, 0] = raw[i, 4 + 2 * k] / 128.0 * aw + ax
            kps[k, 1] = raw[i, 5 + 2 * k] / 128.0 * ah + ay
        out.append(Detection(float(scores[i]), box, kps))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def weighted_nms(dets: List[Detection], threshold: float = 0.3) -> List[Detection]:
    """对应 NonMaxSuppressionCalculator 的 WEIGHTED 算法。"""
    remaining = sorted(dets, key=lambda d: d.score, reverse=True)
    out: List[Detection] = []
    while remaining:
        best = remaining[0]
        candidates = [d for d in remaining if _iou(best.box, d.box) > threshold]
        remaining = [d for d in remaining if _iou(best.box, d.box) <= threshold]
        total = sum(d.score for d in candidates)
        if total > 0:
            box = np.zeros(4, dtype=np.float32)
            kps = np.zeros((6, 2), dtype=np.float32)
            for d in candidates:
                box += d.box * d.score
                kps += d.keypoints * d.score
            out.append(Detection(best.score, box / total, kps / total))
        else:
            out.append(best)
    return out


# ---------------------------------------------------------------------------
# ImageToTensor（对应 image_to_tensor_converter_opencv）
# ---------------------------------------------------------------------------

def image_to_tensor(image_bgr: np.ndarray, cx: float, cy: float, w: float, h: float,
                    rotation: float, out_size: int, range_min: float, range_max: float) -> np.ndarray:
    """旋转子矩形 → out_size×out_size 张量。cx/cy/w/h 为绝对像素。返回 RGB float32 NHWC。"""
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    dx, dy = w / 2.0, h / 2.0
    corners = np.array([
        [cx - dx * cos_r + dy * sin_r, cy - dx * sin_r - dy * cos_r],  # 左上
        [cx + dx * cos_r + dy * sin_r, cy + dx * sin_r - dy * cos_r],  # 右上
        [cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r],  # 右下
        [cx - dx * cos_r - dy * sin_r, cy - dx * sin_r + dy * cos_r],  # 左下
    ], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    crop = cv2.warpPerspective(image_bgr, m, (out_size, out_size),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    scale = (range_max - range_min) / 255.0
    return (rgb * scale + range_min)[None, ...]


# ---------------------------------------------------------------------------
# 完整流水线
# ---------------------------------------------------------------------------

class OnnxFaceLandmarker:
    def __init__(self, detector_path: str, landmark_path: str,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 num_faces: int = 4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.det = ort.InferenceSession(detector_path, so, providers=["CPUExecutionProvider"])
        self.lmk = ort.InferenceSession(landmark_path, so, providers=["CPUExecutionProvider"])
        self.anchors = build_blazeface_anchors()
        self.min_det = float(min_detection_confidence)
        self.min_presence = float(min_presence_confidence)
        self.num_faces = int(num_faces)

    def detect_faces(self, image_bgr: np.ndarray) -> List[Detection]:
        ih, iw = image_bgr.shape[:2]
        side = float(max(iw, ih))  # keep_aspect_ratio letterbox：以长边为方形 ROI
        tensor = image_to_tensor(image_bgr, iw / 2.0, ih / 2.0, side, side, 0.0, 128, -1.0, 1.0)
        reg, cls = self.det.run(["regressors", "classificators"], {"input": tensor})
        dets = decode_detections(reg, cls, self.anchors, self.min_det)
        dets = weighted_nms(dets, 0.3)
        # tensor 归一化坐标 → 图像归一化坐标（去 letterbox）
        for d in dets:
            for arr in (d.box.reshape(-1, 2), d.keypoints):
                arr[:, 0] = (arr[:, 0] - 0.5) * side / iw + 0.5
                arr[:, 1] = (arr[:, 1] - 0.5) * side / ih + 0.5
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.num_faces]

    def detect(self, image_bgr: np.ndarray):
        """返回 [(landmarks_478x3 归一化, presence), ...]，与 MediaPipe Tasks 输出语义一致。"""
        ih, iw = image_bgr.shape[:2]
        results = []
        for det in self.detect_faces(image_bgr):
            # DetectionsToRectsCalculator：检测框 → rect，旋转角由双眼连线确定
            x0, y0, x1, y1 = det.box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            w, h = (x1 - x0), (y1 - y0)
            re_x, re_y = det.keypoints[0]  # keypoint 0: 右眼
            le_x, le_y = det.keypoints[1]  # keypoint 1: 左眼
            rotation = _normalize_radians(
                0.0 - math.atan2(-(le_y * ih - re_y * ih), le_x * iw - re_x * iw))
            # RectTransformation：square_long + scale 1.5
            long_side = max(w * iw, h * ih)
            rw, rh = long_side * 1.5, long_side * 1.5
            tensor = image_to_tensor(image_bgr, cx * iw, cy * ih, rw, rh, rotation, 256, 0.0, 1.0)
            lm_raw, flag_raw, _ = self.lmk.run(
                ["Identity", "Identity_1", "Identity_2"], {"input_12": tensor})
            presence = 1.0 / (1.0 + math.exp(-float(flag_raw.reshape(-1)[0])))
            if presence < self.min_presence:
                continue
            lm = lm_raw.reshape(478, 3).astype(np.float32) / 256.0
            # LandmarkProjection：crop 归一化坐标 → 图像归一化坐标
            out = np.zeros_like(lm)
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            x_c, y_c = lm[:, 0] - 0.5, lm[:, 1] - 0.5
            out[:, 0] = (x_c * cos_r - y_c * sin_r) * (rw / iw) + cx
            out[:, 1] = (x_c * sin_r + y_c * cos_r) * (rh / ih) + cy
            out[:, 2] = lm[:, 2] * (rw / iw)
            results.append((out, presence))
        return results


def _normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector", default="/tmp/task_unzip/face_detector.onnx")
    ap.add_argument("--landmarker", default="/tmp/task_unzip/face_landmarks_detector.onnx")
    ap.add_argument("--golden", default=None, help="golden landmarks txt（原图坐标系）")
    ap.add_argument("--scale", type=float, default=0.25, help="人脸检测处理比例（常用模式 0.25）")
    args = ap.parse_args()

    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    model = OnnxFaceLandmarker(args.detector, args.landmarker)
    small = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                       interpolation=cv2.INTER_AREA) if args.scale < 0.999 else img
    faces = model.detect(small)
    print(f"检测到 {len(faces)} 张人脸")
    if not faces:
        raise SystemExit(1)
    lm_norm, presence = faces[0]
    h, w = small.shape[:2]
    lm_px = lm_norm.copy()
    lm_px[:, 0] *= w
    lm_px[:, 1] *= h
    lm_px[:, 2] *= max(w, h)
    lm_px /= args.scale  # 映射回原图坐标
    print(f"presence={presence:.4f} 鼻尖(idx4)={lm_px[4, :2]}")

    if args.golden:
        lines = [l for l in open(args.golden, encoding="utf-8") if not l.startswith("#")]
        n = int(lines[0])
        g = np.array([[float(v) for v in l.split()] for l in lines[1:1 + n]], dtype=np.float32)
        d = np.linalg.norm(lm_px[:, :2] - g[:, :2], axis=1)
        print(f"对比 golden：mean={d.mean():.4f}px max={d.max():.4f}px 鼻尖={d[4]:.4f}px")
