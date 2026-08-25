# -*- coding: utf-8 -*-
"""facehi 单一 ONNX 模型的傻瓜调用封装。

用户最终只需要两个文件：
- onnx/models/facehi.onnx           唯一对外模型（内嵌子模型与配置）
- libfacehi_custom_ops.so / .dll    ORT 自定义算子库（必须注册，ORT 设计如此）

最简用法::

    from facehi_onnx import remove_highlight
    out_bgr = remove_highlight("photo.png")                # 返回 BGR ndarray
    remove_highlight("photo.png", save_to="photo_out.png") # 直接落盘

底层等价于::

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.register_custom_ops_library("libfacehi_custom_ops.so")
    sess = ort.InferenceSession("onnx/models/facehi.onnx", so)
    result, mask = sess.run(None, {"image": bgr_uint8_hwc})
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = ["FacehiOnnx", "remove_highlight"]

_LIB_NAMES = ("libfacehi_custom_ops.so", "libfacehi_custom_ops.dylib",
              "facehi_custom_ops.dll", "libfacehi_custom_ops.dll")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_model() -> Path | None:
    env = os.environ.get("FACEHI_ONNX_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    cand = _repo_root() / "onnx" / "models" / "facehi.onnx"
    if cand.is_file():
        return cand
    cand = Path(__file__).resolve().parent / "facehi.onnx"
    return cand if cand.is_file() else None


def _find_ops_lib() -> Path | None:
    env = os.environ.get("FACEHI_ORT_CUSTOM_OPS")
    if env and Path(env).is_file():
        return Path(env)
    root = _repo_root()
    for base in (root / "cpp" / "build", root / "onnx" / "lib",
                 Path(__file__).resolve().parent):
        for name in _LIB_NAMES:
            cand = base / name
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


class FacehiOnnx:
    """持有 facehi.onnx 会话；可复用以摊薄模型加载时间。"""

    def __init__(self, model_path=None, ops_lib=None, intra_op_threads: int = 0):
        import onnxruntime as ort

        model = Path(model_path) if model_path else _find_model()
        lib = Path(ops_lib) if ops_lib else _find_ops_lib()
        if model is None or not model.is_file():
            raise FileNotFoundError(
                "找不到 facehi.onnx，请设置 FACEHI_ONNX_MODEL 或传入 model_path"
                "（生成方法见 onnx/README.md）")
        if lib is None or not lib.is_file():
            raise FileNotFoundError(
                "找不到 libfacehi_custom_ops 自定义算子库，请设置 FACEHI_ORT_CUSTOM_OPS "
                "或传入 ops_lib（构建方法见 onnx/README.md）。"
                "纯 onnxruntime 不注册该库无法运行 ai.facehi 自定义域，这是 ORT 的设计。")
        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.register_custom_ops_library(str(lib))
        self.model_path = model
        self.ops_lib = lib
        self.session = ort.InferenceSession(str(model), so,
                                            providers=["CPUExecutionProvider"])

    def run(self, image, landmarks: np.ndarray | None = None):
        """返回 (result_bgr, highlight_mask)，均为 uint8 ndarray。

        landmarks: 可选 [478,3] float32 关键点（注入后跳过内置人脸检测，
        用于黄金对照或复用外部关键点）。
        """
        bgr = _read_bgr(image)
        feeds = {"image": bgr}
        if landmarks is not None:
            feeds["landmarks"] = np.ascontiguousarray(landmarks, dtype=np.float32)
        result, mask = self.session.run(["result", "highlight_mask"], feeds)
        return result, mask


_default_session: FacehiOnnx | None = None


def remove_highlight(image, save_to=None, return_mask: bool = False,
                     model_path=None, ops_lib=None):
    """一行调用去高光。image 可以是图片路径或 uint8 BGR ndarray。

    返回结果 BGR ndarray；return_mask=True 时返回 (result, highlight_mask)。
    save_to 提供时把结果 PNG/JPG 写盘。
    """
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
