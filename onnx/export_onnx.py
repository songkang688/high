# -*- coding: utf-8 -*-
"""从 models/face_landmarker.task 拆出 TFLite 并转换为 ONNX。

face_landmarker.task 本质是一个 zip，内含：
- face_detector.tflite            BlazeFace short-range 人脸检测（128x128, 896 anchors）
- face_landmarks_detector.tflite  478 点人脸关键点（256x256, 输出 1434=478*3）
- face_blendshapes.tflite         表情系数（本工程不使用，不转换）

转换后模型保存到 onnx/models/。转换等价性验证（同一随机输入）：
- face_detector: max abs diff ~6e-5（raw logits）
- face_landmarks_detector: max abs diff ~4e-4（0..256 像素坐标，相对误差 ~1e-6）

依赖：pip install tf2onnx tensorflow-cpu onnxruntime
用法：.venv/bin/python onnx/export_onnx.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "models" / "face_landmarker.task"
OUT_DIR = ROOT / "onnx" / "models"


def main() -> int:
    if not TASK.is_file():
        print(f"[错误] 找不到 {TASK}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(TASK) as zf:
            zf.extractall(tmp_dir)
        for name in ["face_detector", "face_landmarks_detector"]:
            tflite = tmp_dir / f"{name}.tflite"
            onnx_out = OUT_DIR / f"{name}.onnx"
            print(f"[转换] {name}.tflite → {onnx_out}")
            subprocess.run(
                [sys.executable, "-m", "tf2onnx.convert",
                 "--tflite", str(tflite), "--output", str(onnx_out), "--opset", "13"],
                check=True,
            )
    print("[完成] ONNX 模型已生成到 onnx/models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
