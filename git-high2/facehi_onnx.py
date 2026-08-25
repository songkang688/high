# -*- coding: utf-8 -*-
"""git-high2 傻瓜调用封装：facehi*.onnx 单一模型 + 自定义算子库。

本目录（git-high2/）自包含运行 ONNX 版所需的一切：
- models/facehi.onnx                常用档（强力向；内嵌子模型与配置，约 6.3MB）
- models/facehi_daily.onnx          日常/平衡档（烘焙 高保真模式：检测=正常、
                                    修复=正常、method=混合、process_scale=compromise，
                                    详见 models/DAILY.md）
- lib/libfacehi_custom_ops.so       ORT 自定义算子库（两个模型共用同一内核；
                                    必须注册，ORT 设计如此；未提供时可按
                                    README.md / cpp/ 里的说明自行编译）

最简用法::

    from facehi_onnx import remove_highlight
    out_bgr = remove_highlight("photo.png")                # 常用档，返回 BGR ndarray
    remove_highlight("photo.png", save_to="photo_out.png") # 直接落盘
    remove_highlight("photo.png", variant="daily")         # 日常/平衡档

底层等价于::

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.register_custom_ops_library("lib/libfacehi_custom_ops.so")
    sess = ort.InferenceSession("models/facehi.onnx", so)
    result, mask = sess.run(None, {"image": bgr_uint8_hwc})
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = ["FacehiOnnx", "remove_highlight", "find_model", "find_ops_lib",
           "MODEL_VARIANTS"]

_HERE = Path(__file__).resolve().parent
_LIB_NAMES = ("libfacehi_custom_ops.so", "libfacehi_custom_ops.dylib",
              "facehi_custom_ops.dll", "libfacehi_custom_ops.dll")

# 模型档位：None=常用（强力向，facehi.onnx）；"daily"=日常/平衡（facehi_daily.onnx，
# 烘焙 高保真模式，见 models/DAILY.md）。所有档位共用同一自定义算子库。
MODEL_VARIANTS = {None: "facehi.onnx", "daily": "facehi_daily.onnx"}


def find_model(variant: str | None = None) -> Path | None:
    """定位模型文件。variant=None 时环境变量 FACEHI_ONNX_MODEL 优先，
    其次 git-high2/models/facehi.onnx；variant="daily" 定位 facehi_daily.onnx。"""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"未知模型档位: {variant}（可选: {sorted(k for k in MODEL_VARIANTS if k)}）")
    if variant is None:
        env = os.environ.get("FACEHI_ONNX_MODEL")
        if env and Path(env).is_file():
            return Path(env)
    name = MODEL_VARIANTS[variant]
    for cand in (_HERE / "models" / name, _HERE / name):
        if cand.is_file():
            return cand
    return None


def find_ops_lib() -> Path | None:
    """定位自定义算子库：
    环境变量 FACEHI_ORT_CUSTOM_OPS → git-high2/lib/ → git-high2/cpp/build/
    → 向上找仓库根的 cpp/build/（在仓库内运行时复用已有构建产物）。
    """
    env = os.environ.get("FACEHI_ORT_CUSTOM_OPS")
    if env and Path(env).is_file():
        return Path(env)
    bases = [_HERE / "lib", _HERE / "cpp" / "build", _HERE]
    for parent in _HERE.parents:
        bases.append(parent / "cpp" / "build")
        if (parent / ".git").exists():
            break
    for base in bases:
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
        # np.fromfile + imdecode：Windows / 中文路径安全
        data = np.fromfile(str(image), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"图片读取失败: {image}")
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError("需要 uint8 BGR [H,W,3] 图像")
    return np.ascontiguousarray(arr)


class FacehiOnnx:
    """持有 facehi*.onnx 会话；可复用以摊薄模型加载时间。

    variant=None 加载常用档 facehi.onnx；variant="daily" 加载日常/平衡档
    facehi_daily.onnx（烘焙 高保真模式）。model_path 显式给出时忽略 variant。
    """

    def __init__(self, model_path=None, ops_lib=None, intra_op_threads: int = 0,
                 variant: str | None = None):
        import onnxruntime as ort

        model = Path(model_path) if model_path else find_model(variant)
        lib = Path(ops_lib) if ops_lib else find_ops_lib()
        if model is None or not model.is_file():
            expect = MODEL_VARIANTS.get(variant, "facehi.onnx")
            raise FileNotFoundError(
                f"找不到 {expect}，请设置 FACEHI_ONNX_MODEL 或传入 model_path"
                f"（正常情况下应位于 git-high2/models/{expect}）")
        if lib is None or not lib.is_file():
            raise FileNotFoundError(
                "找不到 libfacehi_custom_ops 自定义算子库，请设置 FACEHI_ORT_CUSTOM_OPS "
                "或传入 ops_lib（编译方法见 git-high2/README.md）。"
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


_default_sessions: dict = {}


def remove_highlight(image, save_to=None, return_mask: bool = False,
                     model_path=None, ops_lib=None, variant: str | None = None):
    """一行调用去高光。image 可以是图片路径或 uint8 BGR ndarray。

    variant=None 用常用档 facehi.onnx；variant="daily" 用日常/平衡档
    facehi_daily.onnx（烘焙 高保真模式，去高光够用且更收敛，见 models/DAILY.md）。
    返回结果 BGR ndarray；return_mask=True 时返回 (result, highlight_mask)。
    save_to 提供时把结果 PNG/JPG 写盘（中文路径安全）。
    """
    if model_path or ops_lib:
        sess = FacehiOnnx(model_path=model_path, ops_lib=ops_lib, variant=variant)
    else:
        if variant not in _default_sessions:
            _default_sessions[variant] = FacehiOnnx(variant=variant)
        sess = _default_sessions[variant]
    result, mask = sess.run(image)
    if save_to:
        import cv2

        ext = Path(save_to).suffix or ".png"
        ok, buf = cv2.imencode(ext, result)
        if not ok:
            raise IOError(f"结果编码失败: {save_to}")
        buf.tofile(str(save_to))
    return (result, mask) if return_mask else result
