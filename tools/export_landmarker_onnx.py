"""从 models/face_landmarker.task 导出 ONNX 模型。

MediaPipe 的 .task 是一个 zip 包，内含：
  - face_detector.tflite            BlazeFace 短距人脸检测（128x128，896 anchors）
  - face_landmarks_detector.tflite  FaceMesh-V2 478 点关键点（256x256）
  - face_blendshapes.tflite         表情系数（本流程不使用，不导出）

两个被导出的模型均只包含 TFLite 标准内建算子（无 MediaPipe 自定义算子），
可通过 tf2onnx 无损转换为 ONNX，并用随机输入对拍 TFLite 与 ONNX Runtime 的输出差异。

用法：
  pip install tensorflow-cpu tf2onnx onnx onnxruntime
  python tools/export_landmarker_onnx.py

输出：
  models/face_detector.onnx
  models/face_landmarks_detector.onnx
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "models" / "face_landmarker.task"
EXPORTS = {
    "face_detector.tflite": ("face_detector.onnx", (1, 128, 128, 3)),
    "face_landmarks_detector.tflite": ("face_landmarks_detector.onnx", (1, 256, 256, 3)),
}
OPSET = 17


def convert(tflite_path: Path, onnx_path: Path) -> None:
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", str(OPSET),
    ]
    subprocess.run(cmd, check=True)


def verify(tflite_path: Path, onnx_path: Path, input_shape) -> float:
    """随机输入对拍 TFLite 解释器与 ONNX Runtime，返回全输出最大绝对差。"""
    import onnxruntime as ort
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        x = rng.uniform(0.0, 1.0, input_shape).astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        tfl = [interpreter.get_tensor(o["index"]) for o in interpreter.get_output_details()]
        ort_out = sess.run(None, {sess.get_inputs()[0].name: x})
        for v in ort_out:
            # tflite 输出顺序可能不同，按形状匹配后取最优
            diffs = [
                float(np.abs(t.astype(np.float64) - v.astype(np.float64)).max())
                for t in tfl if t.shape == v.shape
            ]
            worst = max(worst, min(diffs))
    return worst


def main() -> int:
    if not TASK_PATH.is_file():
        print(f"[错误] 找不到 {TASK_PATH}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(TASK_PATH) as z:
            names = set(z.namelist())
            for tflite_name in EXPORTS:
                if tflite_name not in names:
                    print(f"[错误] task 包内缺少 {tflite_name}", file=sys.stderr)
                    return 1
            z.extractall(tmp_dir)
        for tflite_name, (onnx_name, shape) in EXPORTS.items():
            onnx_path = ROOT / "models" / onnx_name
            convert(tmp_dir / tflite_name, onnx_path)
            diff = verify(tmp_dir / tflite_name, onnx_path, shape)
            print(f"[完成] {onnx_name}: TFLite vs ONNX 最大输出差 = {diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
