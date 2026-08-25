"""单 ONNX 模型（models/high_removal.onnx + libhigh_removal_ops.so）的进程内共享会话。

调用方式与 tools/run_high_onnx.py 完全一致（该脚本刻意保持零仓库依赖、可单独分发，
因此不从这里导入）。本模块给仓库内代码（如 app_studio.py）提供：

  - 算子库路径解析：环境变量 HIGH_OPS_LIB 优先，其次 cpp/build/libhigh_removal_ops.so；
  - 惰性单例 InferenceSession：首次调用时构建，之后复用，避免每次点击重建会话；
  - uint8 HWC BGR 输入 / 输出的推理封装（与 cv2.imread / cv2.imwrite 直接对接）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "high_removal.onnx"
DEFAULT_OPS_PATH = ROOT / "cpp" / "build" / "libhigh_removal_ops.so"
OPS_ENV_VAR = "HIGH_OPS_LIB"

_lock = threading.Lock()
_session = None
_session_key: Optional[Tuple[str, str]] = None


class OnnxEngineUnavailable(RuntimeError):
    """ONNX 引擎不可用（缺算子库 / 模型 / onnxruntime），message 为可直接展示的中文提示。"""


def build_hint() -> str:
    """算子库缺失时的构建指引（与 cpp/README.md 一致）。"""
    return (
        f"未找到自定义算子库（默认路径 {DEFAULT_OPS_PATH}，也可用环境变量 {OPS_ENV_VAR} 指定）。\n"
        "请按 cpp/README.md 构建：\n"
        "  sudo apt install cmake g++ libopencv-dev libyaml-cpp-dev\n"
        "  # 从 https://github.com/microsoft/onnxruntime/releases 下载并解压 onnxruntime-linux-x64-1.22.0.tgz\n"
        "  cd cpp && mkdir -p build && cd build\n"
        "  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \\\n"
        "        -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 ..\n"
        "  make -j$(nproc)"
    )


def resolve_ops_library() -> Optional[Path]:
    """解析算子库路径：HIGH_OPS_LIB 优先，其次仓库默认构建产物；找不到返回 None。"""
    env = os.environ.get(OPS_ENV_VAR, "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.is_file() else None
    return DEFAULT_OPS_PATH if DEFAULT_OPS_PATH.is_file() else None


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
    return True, f"ONNX 引擎就绪（算子库：{ops}）。"


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
