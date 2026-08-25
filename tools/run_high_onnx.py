"""单 ONNX 模型调用示例（用户实际使用的方式，无 Gradio、无本仓库 Python 代码依赖）。

只需要三样东西：
  1. models/high_removal.onnx        —— 单文件模型（内嵌两个人脸网络 + default.yaml 配置）
  2. 自定义算子内核 .so（cpp/ 构建产物，作用等同于 onnxruntime 本体）：
       - libhigh_removal_pyops.so（--kernel exact，默认）：session.run 在本进程内调用
         原始 Python 流水线本体，输出与 highlight_removal.pipeline.process_image
         **逐位相同**；要求本仓库可导入（.so 会按自身位置自动定位仓库根，或设
         HIGH_PY_KERNEL_PATH）且已安装 mediapipe；
       - libhigh_removal_ops.so（--kernel cpp）：完整 C++ 移植，无 Python/MediaPipe
         依赖，存在亚像素级浮点尾差（实测见 tools/PARITY_RESULTS.md）。
  3. pip install onnxruntime opencv-python

核心调用就是这几行（两个内核共用同一个 onnx 文件）：

    so = ort.SessionOptions()
    so.register_custom_ops_library("libhigh_removal_pyops.so")   # 或 libhigh_removal_ops.so
    sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
    result = sess.run(["result"], {"image": bgr_uint8_hwc})[0]

用法：
  python tools/run_high_onnx.py --input data/1.png --output out.png [--mask mask.png]
  python tools/run_high_onnx.py --kernel cpp --input data/1.png --output out.png
  python tools/run_high_onnx.py --ops /path/to/lib.so --input data/1.png --output out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
KERNEL_LIBS = {
    "exact": ROOT / "cpp" / "build" / "libhigh_removal_pyops.so",
    "cpp": ROOT / "cpp" / "build" / "libhigh_removal_ops.so",
}


def create_session(model_path: str | Path, ops_library: str | Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(ops_library))
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def remove_highlight(session: ort.InferenceSession, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """image_bgr: uint8 [H, W, 3]（cv2.imread 的原样输出）→ (result, hard_mask)。"""
    image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    result, hard_mask = session.run(["result", "hard_mask"], {"image": image_bgr})
    return result, hard_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "models" / "high_removal.onnx"))
    parser.add_argument("--kernel", choices=["exact", "cpp"], default="exact",
                        help="内核选择：exact = 精确内核（与 Python 逐位一致，默认）；cpp = C++ 快速内核")
    parser.add_argument("--ops", default=None,
                        help="自定义算子库路径（显式指定时覆盖 --kernel 的默认路径）")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--mask", default=None, help="可选：另存高光硬掩码")
    args = parser.parse_args()

    ops = Path(args.ops) if args.ops else KERNEL_LIBS[args.kernel]
    if not ops.is_file():
        raise SystemExit(f"[错误] 找不到自定义算子库 {ops}，请先按 cpp/README.md 构建")
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"[错误] 图片读取失败: {args.input}")

    session = create_session(args.model, ops)
    result, hard_mask = remove_highlight(session, image)

    if not cv2.imwrite(args.output, result):
        raise SystemExit(f"[错误] 结果保存失败: {args.output}")
    if args.mask:
        cv2.imwrite(args.mask, hard_mask)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
