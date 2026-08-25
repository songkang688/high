"""单 ONNX 模型（models/high_removal.onnx + 自定义算子库）的进程内共享会话。

同一个 ONNX 文件支持两个可互换的内核库（图不需要任何改动，注册哪个库就用哪个内核）：

  - 精确内核 libhigh_removal_pyops.so（Windows：high_removal_pyops.dll，默认）：
    Compute 时在宿主进程内调用原始 Python 流水线（highlight_removal.exact_kernel →
    process_image，MediaPipe + pip OpenCV 本体），session.run 输出与直接调用
    process_image **逐位相同**；
  - C++ 快速内核 libhigh_removal_ops.so（Windows：high_removal_ops.dll）：
    完整 C++ 移植（ORT 关键点 + OpenCV C++），
    不依赖 Python/MediaPipe，可被纯 C++ 宿主加载，但与 Python 存在亚像素级浮点尾差
    （实测见 tools/PARITY_RESULTS.md）。

选择方式：
  - 环境变量 HIGH_ONNX_KERNEL=exact（默认）或 cpp；
  - 环境变量 HIGH_OPS_LIB=/path/to/lib.so 直接指定库文件（优先级最高）。

调用方式与 tools/run_high_onnx.py 完全一致（该脚本刻意保持零仓库依赖、可单独分发，
因此不从这里导入）。本模块给仓库内代码（如 app_studio.py）提供惰性单例
InferenceSession 与 uint8 HWC BGR 输入/输出的推理封装。
"""
from __future__ import annotations

import os
import platform
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "high_removal.onnx"
_WINDOWS = platform.system() == "Windows"
_KERNEL_FILES = {
    "exact": "high_removal_pyops.dll" if _WINDOWS else "libhigh_removal_pyops.so",
    "cpp": "high_removal_ops.dll" if _WINDOWS else "libhigh_removal_ops.so",
}
# 查找顺序：cpp/build（Linux make / 手动放置）→ cpp/build/Release（Windows VS 多配置构建）
# → cpp/build/windows（仓库内预编译 DLL，由 .github/workflows/windows-dll.yml 产出）。
_KERNEL_SEARCH_DIRS = [
    ROOT / "cpp" / "build",
    ROOT / "cpp" / "build" / "Release",
    ROOT / "cpp" / "build" / "windows",
]


def _find_kernel_lib(kernel: str) -> Path:
    """按平台文件名在候选构建目录中查找内核库；找不到时返回默认路径用于报错提示。"""
    name = _KERNEL_FILES[kernel]
    for directory in _KERNEL_SEARCH_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return _KERNEL_SEARCH_DIRS[0] / name


EXACT_OPS_PATH = _find_kernel_lib("exact")
CPP_OPS_PATH = _find_kernel_lib("cpp")
OPS_ENV_VAR = "HIGH_OPS_LIB"
KERNEL_ENV_VAR = "HIGH_ONNX_KERNEL"
KERNEL_EXACT = "exact"
KERNEL_CPP = "cpp"

_lock = threading.Lock()
_session = None
_session_key: Optional[Tuple[str, str]] = None


class OnnxEngineUnavailable(RuntimeError):
    """ONNX 引擎不可用（缺算子库 / 模型 / onnxruntime），message 为可直接展示的中文提示。"""


def selected_kernel() -> str:
    """当前内核选择：环境变量 HIGH_ONNX_KERNEL（exact 默认 / cpp）。"""
    raw = os.environ.get(KERNEL_ENV_VAR, "").strip().lower()
    return KERNEL_CPP if raw == KERNEL_CPP else KERNEL_EXACT


def build_hint() -> str:
    """算子库缺失时的构建指引（与 cpp/README.md 一致）。"""
    return (
        f"未找到自定义算子库（精确内核默认路径 {EXACT_OPS_PATH}，C++ 内核 {CPP_OPS_PATH}；\n"
        f"也可用环境变量 {OPS_ENV_VAR} 指定库文件，{KERNEL_ENV_VAR}=exact/cpp 切换内核）。\n"
        "请按 cpp/README.md 构建（两个库由同一次 cmake/make 产出）：\n"
        "  sudo apt install cmake g++ libopencv-dev libyaml-cpp-dev python3-dev\n"
        "  # 从 https://github.com/microsoft/onnxruntime/releases 下载并解压 onnxruntime-linux-x64-1.22.0.tgz\n"
        "  cd cpp && mkdir -p build && cd build\n"
        "  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \\\n"
        "        -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 ..\n"
        "  make -j$(nproc)"
    )


def resolve_ops_library() -> Optional[Path]:
    """解析算子库路径：HIGH_OPS_LIB 最优先，其次按 HIGH_ONNX_KERNEL 选默认构建产物。"""
    env = os.environ.get(OPS_ENV_VAR, "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.is_file() else None
    default = EXACT_OPS_PATH if selected_kernel() == KERNEL_EXACT else CPP_OPS_PATH
    return default if default.is_file() else None


def _missing_ops_message() -> str:
    env = os.environ.get(OPS_ENV_VAR, "").strip()
    if env:
        return f"环境变量 {OPS_ENV_VAR}={env} 指向的算子库不存在。\n{build_hint()}"
    return build_hint()


def availability() -> Tuple[bool, str]:
    """(是否可用, 中文说明)。只检查文件与依赖，不构建会话。"""
    ops = resolve_ops_library()
    if ops is None:
        return False, _missing_ops_message()
    if not DEFAULT_MODEL_PATH.is_file():
        return False, (
            f"未找到单文件模型 {DEFAULT_MODEL_PATH}，"
            "请运行 python tools/export_high_removal_onnx.py 生成。"
        )
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "未安装 onnxruntime，请执行 pip install onnxruntime。"
    kernel_note = "精确内核（与 Python 流水线逐位一致）" if selected_kernel() == KERNEL_EXACT else "C++ 快速内核"
    return True, f"ONNX 引擎就绪（{kernel_note}，算子库：{ops}）。"


def get_session(model_path: str | Path | None = None, ops_library: str | Path | None = None):
    """惰性单例 InferenceSession；路径变化时重建。不可用时抛 OnnxEngineUnavailable。"""
    global _session, _session_key
    model = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    ops = Path(ops_library) if ops_library else resolve_ops_library()
    if ops is None or not ops.is_file():
        raise OnnxEngineUnavailable(_missing_ops_message())
    if not model.is_file():
        raise OnnxEngineUnavailable(
            f"未找到单文件模型 {model}，请运行 python tools/export_high_removal_onnx.py 生成。"
        )
    key = (str(model), str(ops))
    with _lock:
        if _session is None or _session_key != key:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise OnnxEngineUnavailable("未安装 onnxruntime，请执行 pip install onnxruntime。") from exc
            so = ort.SessionOptions()
            so.register_custom_ops_library(str(ops))
            _session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
            _session_key = key
        return _session


def remove_highlight(image_bgr: np.ndarray, session=None) -> Tuple[np.ndarray, np.ndarray]:
    """image_bgr: uint8 [H, W, 3]（cv2.imread 原样）→ (result uint8 [H,W,3], hard_mask uint8 [H,W])。"""
    sess = session if session is not None else get_session()
    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    result, hard_mask = sess.run(["result", "hard_mask"], {"image": image})
    return result, hard_mask
