# -*- coding: utf-8 -*-
"""git-high2 傻瓜调用封装：facehi*.onnx 单文件模型 + 自定义算子库。

本目录（git-high2/）自包含运行 ONNX 版所需的一切：
- models/facehi.onnx                默认模型（烘焙常用模式，内嵌子模型与配置，约 6.3MB）
- models/facehi_strong.onnx         强力版（去高光最狠、细节保护最少）
- models/facehi_daily.onnx          日常版（另一分支交付；缺文件时该档位不可用）
- models/facehi_detail.onnx         保护细节版（另一分支交付；缺文件时该档位不可用）
- lib/libfacehi_custom_ops.so       ORT 自定义算子库（必须注册，ORT 设计如此；
                                    未提供时可按 README.md / cpp/ 里的说明自行编译）

档位环境变量覆盖：FACEHI_ONNX_MODEL（default）、
FACEHI_ONNX_MODEL_STRONG / _DAILY / _DETAIL（对应档位）。

最简用法::

    from facehi_onnx import remove_highlight
    out_bgr = remove_highlight("photo.png")                  # 默认模型，返回 BGR ndarray
    remove_highlight("photo.png", save_to="photo_out.png")   # 直接落盘
    remove_highlight("photo.png", variant="strong")          # 强力版（facehi_strong.onnx）

底层等价于::

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.register_custom_ops_library("lib/libfacehi_custom_ops.so")
    sess = ort.InferenceSession("models/facehi_strong.onnx", so)   # 或 facehi.onnx
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

# 档位别名 → 模型文件名（四档统一 registry，作为各分支合并基准）。
# "default" 即 models/facehi.onnx（烘焙常用模式）；本分支只交付 strong 的
# 模型文件，daily / detail 文件由各自分支交付——缺文件时 find_model 返回 None。
MODEL_VARIANTS = {
    "default": "facehi.onnx",
    "strong": "facehi_strong.onnx",
    "daily": "facehi_daily.onnx",
    "detail": "facehi_detail.onnx",
}

# 分档环境变量覆盖：default 用 FACEHI_ONNX_MODEL（历史行为不变），
# 其余档位用 FACEHI_ONNX_MODEL_{STRONG,DAILY,DETAIL}。
_MODEL_ENV_VARS = {
    "default": "FACEHI_ONNX_MODEL",
    "strong": "FACEHI_ONNX_MODEL_STRONG",
    "daily": "FACEHI_ONNX_MODEL_DAILY",
    "detail": "FACEHI_ONNX_MODEL_DETAIL",
}


def find_model(variant: str | None = None) -> Path | None:
    """定位模型文件；缺文件返回 None。

    variant 为空视同 "default"。variant 可以是 MODEL_VARIANTS 里的档位别名
    （default / strong / daily / detail），也可以直接给文件名
    （如 "facehi_strong.onnx"），在 git-high2/models/ 下查找。
    档位别名支持环境变量覆盖（见 _MODEL_ENV_VARS），环境变量优先于本地文件。
    """
    key = variant or "default"
    env_name = _MODEL_ENV_VARS.get(key)
    if env_name:
        env = os.environ.get(env_name)
        if env and Path(env).is_file():
            return Path(env)
    name = MODEL_VARIANTS.get(key, key)
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

    model_path 直接给模型文件路径；或用 variant 按档位别名 / 文件名在
    git-high2/models/ 下选择（如 variant="strong" → facehi_strong.onnx）。
    """

    def __init__(self, model_path=None, ops_lib=None, intra_op_threads: int = 0,
                 variant: str | None = None):
        import onnxruntime as ort

        model = Path(model_path) if model_path else find_model(variant)
        lib = Path(ops_lib) if ops_lib else find_ops_lib()
        if model is None or not model.is_file():
            key = variant or "default"
            env_name = _MODEL_ENV_VARS.get(key, "FACEHI_ONNX_MODEL")
            name = MODEL_VARIANTS.get(key, key)
            raise FileNotFoundError(
                f"找不到模型（variant={key}，期望 git-high2/models/{name}）。"
                f"该档位文件可能未随本分支交付；请补齐文件、设置环境变量 {env_name} "
                "指向模型，或传入 model_path。")
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


_sessions: dict[str, FacehiOnnx] = {}


def remove_highlight(image, save_to=None, return_mask: bool = False,
                     model_path=None, ops_lib=None, variant: str | None = None):
    """一行调用去高光。image 可以是图片路径或 uint8 BGR ndarray。

    variant 选档位模型（如 "strong" → facehi_strong.onnx），会话按档位缓存复用。
    返回结果 BGR ndarray；return_mask=True 时返回 (result, highlight_mask)。
    save_to 提供时把结果 PNG/JPG 写盘（中文路径安全）。
    """
    if model_path or ops_lib:
        sess = FacehiOnnx(model_path=model_path, ops_lib=ops_lib, variant=variant)
    else:
        key = variant or "default"
        if key not in _sessions:
            _sessions[key] = FacehiOnnx(variant=variant)
        sess = _sessions[key]
    result, mask = sess.run(image)
    if save_to:
        import cv2

        ext = Path(save_to).suffix or ".png"
        ok, buf = cv2.imencode(ext, result)
        if not ok:
            raise IOError(f"结果编码失败: {save_to}")
        buf.tofile(str(save_to))
    return (result, mask) if return_mask else result
