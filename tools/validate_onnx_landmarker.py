"""对拍 ONNX 参考实现与 MediaPipe FaceLandmarker 的 478 点输出。

流水线在 process_scale=compromise 下以 1/4 分辨率检测人脸，
因此同时在原分辨率与 1/4 分辨率上做对拍。

用法：python tools/validate_onnx_landmarker.py [图片... 默认 data/1.png data/2.png data/3.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from onnx_landmarker_ref import OnnxFaceLandmarker  # noqa: E402


def mediapipe_landmarks(image_bgr: np.ndarray):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = image_bgr.shape[:2]
    out = []
    for lms in res.face_landmarks:
        arr = np.array([[p.x * w, p.y * h, p.z * max(w, h)] for p in lms], dtype=np.float64)
        out.append(arr)
    return out


def main() -> int:
    images = sys.argv[1:] or ["data/1.png", "data/2.png", "data/3.png"]
    ours = OnnxFaceLandmarker(ROOT / "models")
    worst = 0.0
    for name in images:
        img = cv2.imread(str(ROOT / name))
        if img is None:
            print(f"[跳过] 读不到 {name}")
            continue
        for scale_label, scale in (("1/1", 1.0), ("1/4", 0.25)):
            small = img if scale >= 0.999 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mp_faces = mediapipe_landmarks(small)
            our_faces = ours.detect_pixels(small)
            if len(mp_faces) != len(our_faces):
                print(f"{name} @{scale_label}: 人脸数不一致 mediapipe={len(mp_faces)} onnx={len(our_faces)}")
                worst = max(worst, 1e9)
                continue
            for i, (mp_lm, (our_lm, score)) in enumerate(zip(mp_faces, our_faces)):
                dxy = np.abs(mp_lm[:, :2] - our_lm[:, :2])
                dmax = float(dxy.max())
                dmean = float(dxy.mean())
                worst = max(worst, dmax)
                print(
                    f"{name} @{scale_label} face{i}: xy 最大偏差 {dmax:.4f}px, 平均 {dmean:.4f}px, presence={score:.4f}"
                )
    print(f"\n全部对拍最大 xy 偏差: {worst:.4f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
