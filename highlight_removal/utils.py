from __future__ import annotations
import os
import json
import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runtime_bootstrap import (
    DEFAULT_RUNTIME_MODE,
    FORCE_CPU_ONLY,
    logical_cpu_count,
    normalize_runtime_mode,
)

import cv2
import numpy as np

_RUNTIME_MODE_NAMES = {
    "CPU正常模式",
    "CPU狂暴模式",
    "CPU低级模式（单线程）",
}


def _logical_cpu_count() -> int:
    return logical_cpu_count()


def _set_env_threads(threads: int) -> None:
    v = str(max(1, int(threads)))
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ]:
        os.environ[key] = v


def _windows_affinity_mask(threads: int) -> int:
    # Windows 单个进程 affinity mask 在普通桌面机器上通常覆盖 64 以内逻辑核。
    n = max(1, min(int(threads), 63))
    return (1 << n) - 1


def _try_set_process_policy(mode: str, threads: int) -> Dict[str, str]:
    """尽量让 CPU 模式真正生效：低级模式限制到 1 核，狂暴模式提升优先级。

    这些设置不是算法正确性的必要条件，失败时不抛异常，只返回中文状态。
    """
    priority_status = "未调整"
    affinity_status = "未调整"
    try:
        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            NORMAL_PRIORITY_CLASS = 0x00000020
            HIGH_PRIORITY_CLASS = 0x00000080
            if mode == "CPU狂暴模式":
                ok = kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
                priority_status = "已提升" if ok else "提升失败"
            elif mode == "CPU低级模式（单线程）":
                ok = kernel32.SetPriorityClass(handle, NORMAL_PRIORITY_CLASS)
                priority_status = "普通优先级" if ok else "设置失败"
            else:
                ok = kernel32.SetPriorityClass(handle, NORMAL_PRIORITY_CLASS)
                priority_status = "普通优先级" if ok else "设置失败"

            # 低级模式强制 1 核；正常/狂暴恢复到 threads 个逻辑核。
            if mode == "CPU低级模式（单线程）":
                mask = 1
            else:
                mask = _windows_affinity_mask(threads)
            ok = kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
            affinity_status = f"已限制到 {threads} 线程掩码" if ok else "设置失败"
        else:
            if hasattr(os, "sched_setaffinity"):
                n = max(1, int(threads))
                cpus = set(range(min(n, _logical_cpu_count())))
                os.sched_setaffinity(0, cpus)
                affinity_status = f"已限制到 {len(cpus)} 个逻辑核"
            try:
                if mode == "CPU狂暴模式":
                    os.nice(-5)
                    priority_status = "已尝试提升"
                else:
                    priority_status = "未提升"
            except Exception:
                priority_status = "无权限提升"
    except Exception as exc:
        priority_status = f"未调整：{exc}"
        affinity_status = f"未调整：{exc}"
    return {"priority": priority_status, "affinity": affinity_status}


def detect_gpu_capabilities() -> Dict[str, Any]:
    """检测 GPU 能力，并说明本工程哪些部分可能真正使用 GPU。"""
    caps: Dict[str, Any] = {
        "opencv_cuda_devices": 0,
        "opencv_cuda_available": False,
        "torch_cuda_available": False,
        "onnxruntime_cuda_available": False,
        "mediapipe_gpu_delegate_possible": False,
        "gpu_name": "",
        "available": False,
        "effective_note": "",
    }
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            n = int(cv2.cuda.getCudaEnabledDeviceCount())
            caps["opencv_cuda_devices"] = n
            caps["opencv_cuda_available"] = n > 0
            if n > 0:
                caps["gpu_name"] = f"OpenCV CUDA 设备 x {n}"
    except Exception:
        pass
    try:
        import torch  # pylint: disable=import-outside-toplevel
        caps["torch_cuda_available"] = bool(torch.cuda.is_available())
        if caps["torch_cuda_available"] and not caps["gpu_name"]:
            try:
                caps["gpu_name"] = str(torch.cuda.get_device_name(0))
            except Exception:
                caps["gpu_name"] = "PyTorch CUDA GPU"
    except Exception:
        pass
    try:
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel
        caps["onnxruntime_cuda_available"] = "CUDAExecutionProvider" in ort.get_available_providers()
        if caps["onnxruntime_cuda_available"] and not caps["gpu_name"]:
            caps["gpu_name"] = "ONNXRuntime CUDA 设备"
    except Exception:
        pass
    try:
        from mediapipe.tasks import python as mp_python  # pylint: disable=import-outside-toplevel
        caps["mediapipe_gpu_delegate_possible"] = hasattr(mp_python.BaseOptions, "Delegate")
    except Exception:
        pass

    caps["available"] = bool(
        caps["opencv_cuda_available"]
        or caps["torch_cuda_available"]
        or caps["onnxruntime_cuda_available"]
        or caps["mediapipe_gpu_delegate_possible"]
    )
    if caps["opencv_cuda_available"]:
        caps["effective_note"] = "OpenCV CUDA 可用；可作为后续核心图像算子 GPU 化基础"
    elif caps["mediapipe_gpu_delegate_possible"]:
        caps["effective_note"] = "MediaPipe Tasks 可能可用 GPU delegate；核心去高光仍主要是 CPU/OpenCV"
    elif caps["torch_cuda_available"] or caps["onnxruntime_cuda_available"]:
        caps["effective_note"] = "检测到 CUDA 框架，但当前传统 OpenCV/NumPy 去高光核心不会自动使用 PyTorch/ONNX GPU"
    else:
        caps["effective_note"] = "未检测到可被当前工程有效调用的 GPU 后端"
    return caps


def apply_runtime_mode(mode: str = DEFAULT_RUNTIME_MODE, fallback_threads: int = 8) -> Dict[str, Any]:
    mode = normalize_runtime_mode(mode)
    cpu_total = _logical_cpu_count()
    gpu_caps = detect_gpu_capabilities()
    warning = ""

    if mode == "CPU低级模式（单线程）":
        device = "CPU"
        threads = 1
        gpu_enabled = False
        opencv_optimized = False
    elif mode == "CPU狂暴模式":
        device = "CPU"
        threads = cpu_total
        gpu_enabled = False
        opencv_optimized = True
    else:
        device = "CPU"
        threads = cpu_total
        gpu_enabled = False
        opencv_optimized = True

    if FORCE_CPU_ONLY:
        gpu_enabled = False
        device = "CPU"
        if mode == "GPU模式":
            mode = DEFAULT_RUNTIME_MODE
            warning = "GPU 模式已禁用，已自动切换 CPU 狂暴模式"

    threads = max(1, int(threads))
    _set_env_threads(threads)

    try:
        cv2.setNumThreads(threads)
        if hasattr(cv2, "setUseOptimized"):
            cv2.setUseOptimized(bool(opencv_optimized))
        cv_threads = int(cv2.getNumThreads())
        cv_optimized = bool(cv2.useOptimized()) if hasattr(cv2, "useOptimized") else bool(opencv_optimized)
    except Exception:
        cv_threads = threads
        cv_optimized = bool(opencv_optimized)

    policy = _try_set_process_policy(mode, threads)
    return {
        "mode": mode,
        "device": device,
        "threads": threads,
        "opencv_threads": cv_threads,
        "opencv_optimized": cv_optimized,
        "gpu_enabled": False,
        "gpu": False,
        "gpu_caps": gpu_caps,
        "priority": policy.get("priority", "未调整"),
        "affinity": policy.get("affinity", "未调整"),
        "warning": warning,
        "cpu_only_policy": FORCE_CPU_ONLY,
    }


def setup_runtime(num_threads: int = 8) -> Dict[str, Any]:
    return apply_runtime_mode(DEFAULT_RUNTIME_MODE, fallback_threads=int(num_threads))


def format_runtime_status(runtime_info: Dict[str, Any]) -> str:
    info = runtime_info or apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    caps = info.get("gpu_caps") or detect_gpu_capabilities()
    lines = [
        f"当前运行模式：{info.get('mode', DEFAULT_RUNTIME_MODE)}",
        f"当前设备：{info.get('device', 'CPU')}",
        f"当前线程数：{info.get('threads', _logical_cpu_count())}",
        f"OpenCV 实际线程数：{info.get('opencv_threads', info.get('threads', _logical_cpu_count()))}",
        f"OpenCV 优化：{'开启' if info.get('opencv_optimized', True) else '关闭'}",
        f"GPU 加速：关闭（工程强制 CPU-only）",
        f"检测到 GPU：{caps.get('gpu_name') or '无/已屏蔽'}",
        f"OpenCV CUDA 设备数：{caps.get('opencv_cuda_devices', 0)}",
        f"PyTorch CUDA：{'可用' if caps.get('torch_cuda_available') else '不可用'}",
        f"MediaPipe GPU delegate：已禁用",
        f"进程优先级：{info.get('priority', '未调整')}",
        f"CPU 亲和性：{info.get('affinity', '未调整')}",
        f"GPU/加速说明：{caps.get('effective_note', '')}",
    ]
    if info.get("warning"):
        lines.append(f"运行模式提示：{info.get('warning')}")
    return "\n".join(lines)


# 模块导入时设置一次，app.py 会根据前端选择再次动态设置。
RUNTIME_INFO = apply_runtime_mode(DEFAULT_RUNTIME_MODE)

import yaml
from PIL import Image


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recursive_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = recursive_update(out[key], value)
        else:
            out[key] = value
    return out


def pil_to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("输入图片为空")
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_uint8(image_bgr), cv2.COLOR_BGR2RGB)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask 为空")
    if mask.dtype == np.bool_:
        return (mask.astype(np.uint8) * 255)
    if mask.max() <= 1:
        return (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    return mask.astype(np.uint8)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask_to_uint8(mask)
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def kernel_size(radius: int | float) -> int:
    r = max(0, int(round(radius)))
    return 2 * r + 1


def morph(mask: np.ndarray, op: str, radius: int | float) -> np.ndarray:
    r = max(0, int(round(radius)))
    m = mask_to_uint8(mask)
    if r <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    if op == "dilate":
        return cv2.dilate(m, k)
    if op == "erode":
        return cv2.erode(m, k)
    if op == "open":
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if op == "close":
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    raise ValueError(f"未知形态学操作: {op}")


def blur_mask(mask: np.ndarray, radius: int | float) -> np.ndarray:
    r = max(0, int(round(radius)))
    m = mask_to_uint8(mask)
    if r <= 0:
        return m
    k = 2 * r + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def fill_poly_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    if points is None or len(points) < 3:
        return mask
    pts = np.round(points[:, :2]).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def fill_hull_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    if points is None or len(points) < 3:
        return mask
    pts = np.round(points[:, :2]).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def line_mask(shape: Tuple[int, int], points: np.ndarray, thickness: int) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    if points is None or len(points) < 2:
        return mask
    pts = np.round(points[:, :2]).astype(np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    return mask


def ellipse_mask(shape: Tuple[int, int], center: Tuple[float, float], axes: Tuple[float, float], angle: float = 0.0) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    ax1, ax2 = max(1, int(round(axes[0]))), max(1, int(round(axes[1])))
    cv2.ellipse(mask, (cx, cy), (ax1, ax2), float(angle), 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    img = ensure_uint8(image_bgr).copy()
    m = normalize_mask(mask)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay = img.astype(np.float32) * (1 - alpha * m[..., None]) + color_arr * (alpha * m[..., None])
    return overlay.clip(0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    m = mask_to_uint8(mask)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)


def safe_bbox_from_points(points: np.ndarray, image_shape: Tuple[int, int, int] | Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    if points is None or len(points) == 0:
        return (0, 0, 0, 0)
    xy = np.asarray(points)[:, :2]
    x0, y0 = np.floor(xy.min(axis=0)).astype(int)
    x1, y1 = np.ceil(xy.max(axis=0)).astype(int)
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    return (x0, y0, max(0, x1 - x0 + 1), max(0, y1 - y0 + 1))


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def iou_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    denom = bbox_area(a) + bbox_area(b) - inter
    return float(inter / denom) if denom > 0 else 0.0


_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


@lru_cache(maxsize=8)
def _load_cjk_font(size: int):
    from PIL import ImageFont

    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_text_cn(
    image_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    font_size: int = 16,
) -> np.ndarray:
    """在 BGR 图像上绘制中文/混合文本。OpenCV putText 不支持 CJK，会显示为问号。"""
    if not text:
        return image_bgr
    if all(ord(c) < 128 for c in text):
        cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color_bgr, 1, cv2.LINE_AA)
        return image_bgr
    from PIL import Image, ImageDraw

    x, y = int(org[0]), int(org[1])
    font = _load_cjk_font(font_size)
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((x, y), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def clamp_process_scale(scale: float) -> float:
    return float(max(0.25, min(1.0, scale)))


COMPROMISE_PROCESS_SCALES = frozenset({"compromise", "折中", "0.25_0.5"})


def resolve_process_scales(pipeline: Dict[str, Any] | None) -> tuple[float, float, bool]:
    """返回 (face_scale, highlight_scale, is_compromise)。"""
    pipeline = pipeline or {}
    raw = pipeline.get("process_scale", 1.0)
    if raw in COMPROMISE_PROCESS_SCALES or str(raw).lower() in COMPROMISE_PROCESS_SCALES:
        return 0.25, 0.5, True
    scale = clamp_process_scale(float(raw))
    return scale, scale, False


def scale_to_fraction(scale: float) -> str:
    if scale >= 0.999:
        return "1"
    for denom in (2, 4, 8):
        if abs(scale - 1.0 / denom) < 0.02:
            return f"1/{denom}"
    return f"{scale:g}"


def describe_resolution_plan(pipeline: Dict[str, Any] | None) -> Dict[str, str]:
    """根据 pipeline 配置，描述各阶段使用的分辨率比例。"""
    pipeline = pipeline or {}
    face_scale, highlight_scale, is_compromise = resolve_process_scales(pipeline)
    refine = bool(pipeline.get("refine_upsampled_masks", True))
    face = scale_to_fraction(face_scale)

    if is_compromise:
        region = "1/4"
        if refine:
            highlight = "原图"
            refine_txt = "开(全分辨率高光)"
        else:
            highlight = "1/2→原图"
            refine_txt = "关"
    elif highlight_scale >= 0.999:
        region = "1"
        highlight = "1"
        refine_txt = "—"
    elif refine:
        region = face
        highlight = "原图"
        refine_txt = "开(跳过低分辨率高光)"
    else:
        hl = scale_to_fraction(highlight_scale)
        region = hl
        highlight = f"{hl}→原图"
        refine_txt = "关"

    return {
        "face": face,
        "region": region,
        "highlight": highlight,
        "refine": refine_txt,
        "removal": "原图",
    }


def annotate_resolution_badge(image_bgr: np.ndarray, lines: List[str], font_size: int = 22) -> np.ndarray:
    """在图像右下角叠加分辨率说明。"""
    if image_bgr is None or not lines:
        return image_bgr
    img = image_bgr.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    from PIL import ImageDraw

    font = _load_cjk_font(font_size)
    probe = ImageDraw.Draw(Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)))
    max_w = 0
    line_h = font_size + 6
    for line in lines:
        if all(ord(c) < 128 for c in line):
            max_w = max(max_w, int(len(line) * font_size * 0.55))
        else:
            bbox = probe.textbbox((0, 0), line, font=font)
            max_w = max(max_w, bbox[2] - bbox[0])
    pad = 10
    block_h = line_h * len(lines) + pad
    block_w = max_w + pad * 2
    h, w = img.shape[:2]
    x0 = max(0, w - block_w - pad)
    y0 = max(pad, h - block_h - pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (min(w - 1, x0 + block_w), min(h - 1, y0 + block_h)), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
    y = y0 + 5
    for line in lines:
        img = put_text_cn(img, line, (x0 + pad, y), (255, 255, 255), font_size=font_size)
        y += line_h
    return img


def scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
    lm = landmarks.copy()
    lm[:, :2] *= scale
    if lm.shape[1] > 2:
        lm[:, 2] *= scale
    return lm


def resize_for_process(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    scale = clamp_process_scale(scale)
    if scale >= 0.999:
        return image_bgr
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def upsample_mask_hard(mask: np.ndarray, shape: Tuple[int, int, int] | Tuple[int, int], *, dilate: bool = False) -> np.ndarray:
    h, w = shape[:2]
    up = cv2.resize(mask_to_uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST)
    if dilate and int((up > 0).sum()) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        up = cv2.dilate(up, k, iterations=1)
    return up


def upsample_mask_soft(mask: np.ndarray, shape: Tuple[int, int, int] | Tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    up = cv2.resize(mask_to_uint8(mask), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(up, (3, 3), 0)


def scaled_highlight_params(params: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """低分辨率检测时，按面积/长度比例缩放阈值类参数。"""
    scale = clamp_process_scale(scale)
    if scale >= 0.999:
        return params
    out = dict(params)
    base_min = float(out.get("highlight_min_area", 8))
    # 0.25 时若按面积缩放 min_area 会过小，导致低分辨率几乎检不出 seed。
    if scale <= 0.26:
        out["highlight_min_area"] = max(2, int(round(base_min * scale)))
    else:
        out["highlight_min_area"] = max(1, int(round(base_min * scale * scale)))
    for key in ("mask_dilate_radius", "mask_erode_radius", "mask_blur_radius", "morph_close_radius"):
        if key in out:
            base = float(out[key])
            scaled = int(round(base * scale))
            # 下限只对显式启用（base > 0）的形态学半径生效：既保证低分辨率下
            # 请求的膨胀/腐蚀/闭合不会被四舍五入成 0，也不会把显式的 0
            # （如保护细节档 mask_dilate_radius: 0）强制抬成 1。
            floor = 1 if base > 0 and key in ("morph_close_radius", "mask_dilate_radius", "mask_erode_radius") else 0
            if key == "mask_blur_radius":
                floor = 3 if scale <= 0.26 else 0
            out[key] = max(floor, scaled)
    return out


def bbox_from_mask(mask: np.ndarray, shape: Tuple[int, int, int] | Tuple[int, int], padding_ratio: float = 0.12) -> Optional[Tuple[int, int, int, int]]:
    active = mask_to_uint8(mask) > 0
    ys, xs = np.where(active)
    if ys.size == 0:
        return None
    h, w = shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(8, int((x1 - x0 + 1) * padding_ratio))
    pad_y = max(8, int((y1 - y0 + 1) * padding_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def merge_bboxes(*boxes: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[0] + b[2] - 1 for b in valid)
    y1 = max(b[1] + b[3] - 1 for b in valid)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def compute_roi_bbox(
    hard_mask: np.ndarray,
    soft_mask: np.ndarray,
    shape: Tuple[int, int, int] | Tuple[int, int],
    padding_ratio: float = 0.12,
) -> Optional[Tuple[int, int, int, int]]:
    active = (mask_to_uint8(hard_mask) > 0) | (mask_to_uint8(soft_mask) > 0)
    ys, xs = np.where(active)
    if ys.size == 0:
        return None
    h, w = shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(8, int((x1 - x0 + 1) * padding_ratio))
    pad_y = max(8, int((y1 - y0 + 1) * padding_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def write_image(path: str | Path, image_bgr_or_rgb: np.ndarray, rgb: bool = False) -> str:
    """Windows 兼容图像保存。避免 cv2.imwrite 在中文路径/中文文件名下静默失败。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ensure_uint8(image_bgr_or_rgb)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower() or ".png"
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
        ext = ".png"
        path = path.with_suffix(ext)
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"图像编码失败：{path}")
    buffer.tofile(str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"图像保存失败：{path}")
    return str(path.resolve())


def empty_outputs_like(image_bgr: np.ndarray | None, message: str = "") -> Dict[str, Any]:
    if image_bgr is None:
        blank = np.zeros((512, 512, 3), np.uint8)
    else:
        blank = np.zeros_like(image_bgr)
    return {
        "result_bgr": blank,
        "highlight_mask": np.zeros(blank.shape[:2], np.uint8),
        "status": message,
    }
