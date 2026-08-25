# -*- coding: utf-8 -*-
"""git-high2 傻瓜调用封装：facehi*.onnx 单文件模型 + 自定义算子库。

本目录（git-high2/）自包含运行 ONNX 版所需的一切：
- models/facehi.onnx                默认模型（烘焙常用模式，内嵌子模型与配置，约 6.3MB）
- models/facehi_strong.onnx         强力版（去高光最狠、细节保护最少）
- models/facehi_daily.onnx          日常/平衡版（烘焙独立「日常模式」：高光检测与修复
                                    各参数取强力档与保护细节档两端的中值，
                                    method=混合、process_scale=compromise，
                                    详见 models/DAILY.md）
- models/facehi_detail.onnx         保护细节版（去高光最弱、纹理保留最多）
- lib/libfacehi_custom_ops.so       ORT 自定义算子库（四档共用同一内核；必须注册，
                                    ORT 设计如此；未提供时可按 README.md / cpp/
                                    里的说明自行编译）

各档模型文件互相独立，缺某一档文件时其余档不受影响（find_model 返回 None）。

最简用法::

    from facehi_onnx import remove_highlight
    out_bgr = remove_highlight("photo.png")                  # 默认模型，返回 BGR ndarray
    remove_highlight("photo.png", save_to="photo_out.png")   # 直接落盘
    remove_highlight("photo.png", variant="daily")           # 日常/平衡版（facehi_daily.onnx）

底层等价于::

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.register_custom_ops_library("lib/libfacehi_custom_ops.so")
    sess = ort.InferenceSession("models/facehi_daily.onnx", so)   # 或 facehi.onnx 等
    result, mask = sess.run(None, {"image": bgr_uint8_hwc})
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = ["FacehiOnnx", "remove_highlight", "find_model", "find_ops_lib",
           "MODEL_VARIANTS", "VARIANT_ALIASES", "normalize_variant"]

_HERE = Path(__file__).resolve().parent
_LIB_NAMES = ("libfacehi_custom_ops.so", "libfacehi_custom_ops.dylib",
              "facehi_custom_ops.dll", "libfacehi_custom_ops.dll")

# 档位键 → 模型文件名（str → str）。四档共用同一自定义算子库，
# 差别只在烘焙进节点属性的 mode / 配置段。
MODEL_VARIANTS = {
    "default": "facehi.onnx",
    "strong": "facehi_strong.onnx",
    "daily": "facehi_daily.onnx",
    "detail": "facehi_detail.onnx",
}

# 中文别名 → 标准档位键。
VARIANT_ALIASES = {
    "默认": "default",
    "常用": "default",
    "强力": "strong",
    "日常": "daily",
    "平衡": "daily",
    "细节": "detail",
    "保护细节": "detail",
}


def normalize_variant(variant: str | None) -> str:
    """把 None / 中文别名归一成标准档位键；未知值原样返回（可作文件名）。"""
    if not variant:
        return "default"
    return VARIANT_ALIASES.get(variant, variant)


def _variant_env_name(key: str) -> str:
    return "FACEHI_ONNX_MODEL" if key == "default" else f"FACEHI_ONNX_MODEL_{key.upper()}"


def find_model(variant: str | None = None) -> Path | None:
    """定位模型文件；缺文件返回 None（不抛异常）。

    variant 为 None / "default"：环境变量 FACEHI_ONNX_MODEL 优先，
    其次 git-high2/models/facehi.onnx。
    其它档位（"strong" / "daily" / "detail" 或中文别名）：对应环境变量
    FACEHI_ONNX_MODEL_STRONG / _DAILY / _DETAIL 优先，其次
    git-high2/models/<档位文件名>。也可直接传文件名（如 "facehi_daily.onnx"）。
    """
    key = normalize_variant(variant)
    if key in MODEL_VARIANTS:
        env = os.environ.get(_variant_env_name(key))
        if env and Path(env).is_file():
            return Path(env)
        name = MODEL_VARIANTS[key]
    else:
        name = key  # 直接给文件名
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

    model_path 直接给模型文件路径；或用 variant 按档位键 / 中文别名 / 文件名在
    git-high2/models/ 下选择（如 variant="daily" → facehi_daily.onnx）。
    """

    def __init__(self, model_path=None, ops_lib=None, intra_op_threads: int = 0,
                 variant: str | None = None):
        import onnxruntime as ort

        key = normalize_variant(variant)
        model = Path(model_path) if model_path else find_model(key)
        lib = Path(ops_lib) if ops_lib else find_ops_lib()
        if model is None or not model.is_file():
            expect = MODEL_VARIANTS.get(key, key)
            raise FileNotFoundError(
                f"找不到模型 {expect}（variant={key}），请设置环境变量 "
                f"{_variant_env_name(key) if key in MODEL_VARIANTS else 'FACEHI_ONNX_MODEL'} "
                f"或传入 model_path（正常情况下应位于 git-high2/models/{expect}）")
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

    variant 选档位模型（"default" / "strong" / "daily" / "detail" 或中文别名，
    None 视同 "default"），会话按档位键缓存复用。
    返回结果 BGR ndarray；return_mask=True 时返回 (result, highlight_mask)。
    save_to 提供时把结果 PNG/JPG 写盘（中文路径安全）。
    """
    if model_path or ops_lib:
        sess = FacehiOnnx(model_path=model_path, ops_lib=ops_lib, variant=variant)
    else:
        key = normalize_variant(variant)
        if key not in _sessions:
            _sessions[key] = FacehiOnnx(variant=key)
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
