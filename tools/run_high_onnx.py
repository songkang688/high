"""单 ONNX 模型调用示例（用户实际使用的方式，无 Gradio、无本仓库 Python 代码依赖）。

只需要三样东西：
  1. models/high_removal.onnx        —— 单文件模型（内嵌两个人脸网络 + default.yaml 配置）
  2. 自定义算子内核库（cpp/ 构建产物，Linux 为 .so、Windows 为 .dll，作用等同于 onnxruntime 本体）：
       - libhigh_removal_pyops.so / high_removal_pyops.dll（--kernel exact，默认）：
         session.run 在本进程内调用原始 Python 流水线本体，输出与
         highlight_removal.pipeline.process_image **逐位相同**；要求本仓库可导入
         （库会按自身位置自动定位仓库根，或设 HIGH_PY_KERNEL_PATH）且已安装 mediapipe；
       - libhigh_removal_ops.so / high_removal_ops.dll（--kernel cpp）：完整 C++ 移植，
         无 Python/MediaPipe 依赖，存在亚像素级浮点尾差（实测见 tools/PARITY_RESULTS.md）。
  3. pip install onnxruntime opencv-python

核心调用就是这几行（两个内核共用同一个 onnx 文件）：

    so = ort.SessionOptions()
    so.register_custom_ops_library("libhigh_removal_pyops.so")   # Windows: high_removal_pyops.dll
    sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
    result = sess.run(["result"], {"image": bgr_uint8_hwc})[0]

三档去油光强度预设（--preset，三个模型都内嵌各自的 configs/onnx_*.yaml，
实测对比见 tools/THREE_ONNX_AUDIT.md）：
  strong = models/high_removal_strong.onnx（去油光最强）
  daily  = models/high_removal_daily.onnx（= 当前 default.yaml 平衡效果）
  detail = models/high_removal_detail.onnx（保护细节，保留更多皮肤纹理）
不给 --preset / --model 时用 models/high_removal.onnx（默认配置，与 daily 输出逐位相同）。

用法：
  python tools/run_high_onnx.py --input data/1.png --output out.png [--mask mask.png]
  python tools/run_high_onnx.py --preset strong --input data/1.png --output out.png
  python tools/run_high_onnx.py --kernel cpp --input data/1.png --output out.png
  python tools/run_high_onnx.py --ops /path/to/lib.so --input data/1.png --output out.png
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
_WINDOWS = platform.system() == "Windows"
KERNEL_FILES = {
    "exact": "high_removal_pyops.dll" if _WINDOWS else "libhigh_removal_pyops.so",
    "cpp": "high_removal_ops.dll" if _WINDOWS else "libhigh_removal_ops.so",
}
# 查找顺序：cpp/build（Linux make / 手动放置）→ cpp/build/Release（Windows VS 多配置构建）
# → cpp/build/windows（仓库内预编译 DLL，由 .github/workflows/windows-dll.yml 产出）。
KERNEL_SEARCH_DIRS = [
    ROOT / "cpp" / "build",
    ROOT / "cpp" / "build" / "Release",
    ROOT / "cpp" / "build" / "windows",
]


def find_kernel_lib(kernel: str) -> Path:
    """按平台文件名在候选构建目录中查找内核库；找不到时返回默认路径用于报错提示。"""
    name = KERNEL_FILES[kernel]
    for directory in KERNEL_SEARCH_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return KERNEL_SEARCH_DIRS[0] / name


KERNEL_LIBS = {kernel: find_kernel_lib(kernel) for kernel in KERNEL_FILES}

# 三档强度预设 → 单文件模型（各自内嵌 configs/onnx_<preset>.yaml）。
PRESET_MODELS = {
    "strong": ROOT / "models" / "high_removal_strong.onnx",
    "daily": ROOT / "models" / "high_removal_daily.onnx",
    "detail": ROOT / "models" / "high_removal_detail.onnx",
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
    parser.add_argument("--model", default=None,
                        help="单文件模型路径（显式指定时覆盖 --preset；默认 models/high_removal.onnx）")
    parser.add_argument("--preset", choices=sorted(PRESET_MODELS), default=None,
                        help="强度预设：strong = 强力 / daily = 日常 / detail = 保护细节")
    parser.add_argument("--kernel", choices=["exact", "cpp"], default="exact",
                        help="内核选择：exact = 精确内核（与 Python 逐位一致，默认）；cpp = C++ 快速内核")
    parser.add_argument("--ops", default=None,
                        help="自定义算子库路径（显式指定时覆盖 --kernel 的默认路径）")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--mask", default=None, help="可选：另存高光硬掩码")
    args = parser.parse_args()

    if args.model:
        model = Path(args.model)
    elif args.preset:
        model = PRESET_MODELS[args.preset]
    else:
        model = ROOT / "models" / "high_removal.onnx"
    if not model.is_file():
        raise SystemExit(f"[错误] 找不到模型文件: {model}")

    ops = Path(args.ops) if args.ops else KERNEL_LIBS[args.kernel]
    if not ops.is_file():
        searched = "、".join(str(d / KERNEL_FILES[args.kernel]) for d in KERNEL_SEARCH_DIRS)
        raise SystemExit(f"[错误] 找不到自定义算子库（已查找：{searched}），请先按 cpp/README.md 构建")
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"[错误] 图片读取失败: {args.input}")

    session = create_session(model, ops)
    result, hard_mask = remove_highlight(session, image)

    if not cv2.imwrite(args.output, result):
        raise SystemExit(f"[错误] 结果保存失败: {args.output}")
    if args.mask:
        cv2.imwrite(args.mask, hard_mask)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
