# -*- coding: utf-8 -*-
"""output-1.22.0 傻瓜调用：本目录内的 facehi.onnx + ORT 1.22.0 自定义算子库。

本目录自包含运行所需文件：
- facehi.onnx                 用 Python onnx==1.22.0 重新导出
- libfacehi_custom_ops.so     Linux x86-64，按 ORT 1.22.0 头文件重编
- facehi_custom_ops.dll       Windows x64，按 ORT 1.22.0 头文件重编

必须安装：pip install -r requirements.txt（onnxruntime==1.22.0）

最简用法::

    from facehi_onnx import remove_highlight
    out_bgr = remove_highlight("photo.png")
    remove_highlight("photo.png", save_to="photo_out.png")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["FacehiOnnx", "remove_highlight", "find_model", "find_ops_lib"]

_HERE = Path(__file__).resolve().parent
_LIB_NAMES = ("libfacehi_custom_ops.so", "libfacehi_custom_ops.dylib",
              "facehi_custom_ops.dll", "libfacehi_custom_ops.dll")
_REQUIRED_ORT = "1.22.0"


def find_model() -> Path | None:
    env = os.environ.get("FACEHI_ONNX_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    for cand in (_HERE / "facehi.onnx", _HERE / "models" / "facehi.onnx"):
        if cand.is_file():
            return cand
    return None


def find_ops_lib() -> Path | None:
    env = os.environ.get("FACEHI_ORT_CUSTOM_OPS")
    if env and Path(env).is_file():
        return Path(env)
    for name in _LIB_NAMES:
        cand = _HERE / name
        if cand.is_file():
            return cand
    return None


def _read_bgr(image) -> np.ndarray:
    import cv2

    if isinstance(image, np.ndarray):
        arr = image
    else:
        data = np.fromfile(str(image), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"图片读取失败: {image}")
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError("需要 uint8 BGR [H,W,3] 图像")
    return np.ascontiguousarray(arr)


def _check_ort_version() -> None:
    import onnxruntime as ort

    ver = ort.__version__
    if ver != _REQUIRED_ORT and not ver.startswith(_REQUIRED_ORT + "."):
        print(
            f"[警告] 本目录按 onnxruntime=={_REQUIRED_ORT} 编译；"
            f"当前是 {ver}。请执行: pip install onnxruntime=={_REQUIRED_ORT}",
            file=sys.stderr,
        )


class FacehiOnnx:
    """持有 facehi.onnx 会话；可复用以摊薄模型加载时间。"""

    def __init__(self, model_path=None, ops_lib=None, intra_op_threads: int = 0):
        import onnxruntime as ort

        _check_ort_version()
        model = Path(model_path) if model_path else find_model()
        lib = Path(ops_lib) if ops_lib else find_ops_lib()
        if model is None or not model.is_file():
            raise FileNotFoundError(
                f"找不到 facehi.onnx（应位于 {_HERE / 'facehi.onnx'}）")
        if lib is None or not lib.is_file():
            raise FileNotFoundError(
                f"找不到自定义算子库（Linux: {_HERE / 'libfacehi_custom_ops.so'}，"
                f"Windows: {_HERE / 'facehi_custom_ops.dll'}）。"
                "纯 onnxruntime 不注册该库无法运行 ai.facehi 自定义域。")
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.register_custom_ops_library(str(lib))
        self.model_path = model
        self.ops_lib = lib
        self.ort_version = ort.__version__
        self.session = ort.InferenceSession(str(model), so,
                                            providers=["CPUExecutionProvider"])

    def run(self, image, landmarks: np.ndarray | None = None):
        """返回 (result_bgr, highlight_mask)，均为 uint8 ndarray。"""
        bgr = _read_bgr(image)
        feeds = {"image": bgr}
        if landmarks is not None:
            feeds["landmarks"] = np.ascontiguousarray(landmarks, dtype=np.float32)
        result, mask = self.session.run(["result", "highlight_mask"], feeds)
        return result, mask


_default_session: FacehiOnnx | None = None


def remove_highlight(image, save_to=None, return_mask: bool = False,
                     model_path=None, ops_lib=None):
    """一行调用去高光。image 可以是图片路径或 uint8 BGR ndarray。"""
    global _default_session
    if model_path or ops_lib:
        sess = FacehiOnnx(model_path=model_path, ops_lib=ops_lib)
    else:
        if _default_session is None:
            _default_session = FacehiOnnx()
        sess = _default_session
    result, mask = sess.run(image)
    if save_to:
        import cv2

        ext = Path(save_to).suffix or ".png"
        ok, buf = cv2.imencode(ext, result)
        if not ok:
            raise IOError(f"结果编码失败: {save_to}")
        buf.tofile(str(save_to))
    return (result, mask) if return_mask else result
