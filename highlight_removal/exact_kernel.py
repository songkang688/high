"""ai.high:HighlightRemoval 精确内核的 Python 侧实现（100% 等效路径）。

libhigh_removal_pyops.so（cpp/src/py_custom_op.cpp）在宿主进程内通过 CPython C API
调用本模块的 run_from_buffer：输入图与 ONNX 节点属性里内嵌的 config_yaml 原始字节
进来，走的就是仓库原始 Python 流水线（MediaPipe FaceLandmarker + pip OpenCV 的
highlight_removal.pipeline.process_image），因此 session.run 的输出与直接调用
process_image 逐位相同（np.array_equal）——没有第二套实现，不存在近似。

配置构建方式与 app_studio.py 的 studio_config() 完全一致：
  yaml.safe_load(内嵌 default.yaml) + apply_runtime_mode(默认CPU模式) + 关闭可视化。

失败语义与 C++ 内核一致：未检测到人脸时原样返回输入图、掩码全零
（process_image 本身就是这个行为）。
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

import numpy as np

_lock = threading.RLock()
_config_cache: Dict[bytes, Dict[str, Any]] = {}


def _build_config(config_yaml: bytes) -> Dict[str, Any]:
    import yaml

    from .runtime_bootstrap import DEFAULT_RUNTIME_MODE
    from .utils import apply_runtime_mode

    cfg = yaml.safe_load(config_yaml.decode("utf-8")) or {}
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def _get_config(config_yaml: bytes) -> Dict[str, Any]:
    cfg = _config_cache.get(config_yaml)
    if cfg is None:
        cfg = _build_config(config_yaml)
        _config_cache[config_yaml] = cfg
    return cfg


def run_array(image_bgr: np.ndarray, config_yaml: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """uint8 [H,W,3] BGR → (result uint8 [H,W,3], hard_mask uint8 [H,W])。

    就是 highlight_removal.pipeline.process_image 本体；加锁串行化以保护共享的
    MediaPipe FaceLandmarker 实例（与 face_detect 的进程级缓存同一实例）。
    """
    from .pipeline import process_image

    with _lock:
        cfg = _get_config(bytes(config_yaml))
        out = process_image(image_bgr, cfg)
    result = np.ascontiguousarray(out.result_bgr, dtype=np.uint8)
    mask = np.ascontiguousarray(out.highlight_mask, dtype=np.uint8)
    if mask.shape != image_bgr.shape[:2]:
        mask = np.zeros(image_bgr.shape[:2], np.uint8)
    return result, mask


def run_from_buffer(image_bytes: bytes, height: int, width: int, config_yaml: bytes) -> Tuple[bytes, bytes]:
    """C++ 蹦床（libhigh_removal_pyops.so）调用的入口：全部用 bytes 传递，避免依赖 NumPy C API。"""
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(int(height), int(width), 3).copy()
    result, mask = run_array(image, config_yaml)
    return result.tobytes(), mask.tobytes()
