from __future__ import annotations

import os
from highlight_removal.runtime_bootstrap import bootstrap_before_numpy, DEFAULT_RUNTIME_MODE, normalize_runtime_mode

bootstrap_before_numpy()

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import time

import cv2
import gradio as gr
import numpy as np
import yaml
from PIL import Image

from highlight_removal.pipeline import process_image, update_config_from_flat
from highlight_removal.face_detect import preload_face_landmarker
from highlight_removal.utils import (
    annotate_resolution_badge,
    apply_runtime_mode,
    bgr_to_rgb,
    describe_resolution_plan,
    detect_gpu_capabilities,
    format_runtime_status,
    load_yaml,
    pil_to_bgr,
    save_yaml,
    setup_runtime,
    write_image,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
DEFAULT_CONFIG = load_yaml(DEFAULT_CONFIG_PATH)
RUNTIME_MODES = ["CPU狂暴模式", "CPU正常模式", "CPU低级模式（单线程）"]
GPU_CAPABILITIES = detect_gpu_capabilities()
RUNTIME = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
OUTPUT_DIR = ROOT / "runtime_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_SESSION_PATH = OUTPUT_DIR / "user_session.yaml"
DETECTION_SENSITIVITY_PATH = ROOT / "configs" / "detection_sensitivity_presets.yaml"
DETECTION_SENSITIVITY_NAMES = ["正常", "灵敏", "弱"]
DEFAULT_DETECTION_SENSITIVITY = "正常"
REMOVAL_INTENSITY_PATH = ROOT / "configs" / "removal_intensity_presets.yaml"
REMOVAL_INTENSITY_NAMES = ["正常", "强力", "削弱"]
DEFAULT_REMOVAL_INTENSITY = "正常"
DEFAULT_IMAGE_DIR_REL = "../data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_dir_path(dir_text: str | Path | None = None) -> Path:
    raw = str(dir_text or "").strip() or DEFAULT_IMAGE_DIR_REL
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.expanduser().resolve()
    if p.is_file():
        return p.parent
    return p


def _resolve_image_path(path_text: str | Path) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.expanduser().resolve()
    return p


def _normalize_stored_path(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return DEFAULT_IMAGE_DIR_REL
    p = _resolve_image_path(text)
    try:
        rel = os.path.relpath(p, ROOT)
        return rel.replace("\\", "/") if os.sep == "\\" else rel
    except ValueError:
        return text.replace("\\", "/")


def _display_path(path: str | Path) -> str:
    return _normalize_stored_path(path)


def _gradio_serve_path(path: str | Path) -> str:
    """Gradio 组件读取文件需绝对路径；界面与持久化仍用相对路径。"""
    return str(_resolve_image_path(path))


DEFAULT_IMAGE_DIR = _resolve_dir_path(DEFAULT_IMAGE_DIR_REL)


def _empty_batch() -> Dict[str, Any]:
    return {"locked": False, "dir": "", "files": [], "index": 0}


def _list_images_in_dir(dir_path: str | Path) -> List[str]:
    root = _resolve_dir_path(dir_path)
    if not root.is_dir():
        return []

    def _sort_key(p: Path):
        if p.stem.isdigit():
            return (0, int(p.stem), p.suffix.lower())
        return (1, p.name.lower())

    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return [_normalize_stored_path(p) for p in sorted(files, key=_sort_key)]


def _first_image_in_dir(dir_path: str | Path | None = None) -> str:
    if dir_path:
        raw = str(dir_path).strip()
        if raw:
            p = _resolve_image_path(raw)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return _normalize_stored_path(p)
            if p.is_dir():
                files = _list_images_in_dir(p)
                return files[0] if files else ""
    files = _list_images_in_dir(DEFAULT_IMAGE_DIR_REL)
    return files[0] if files else ""


DEFAULT_FIRST_IMAGE = _first_image_in_dir()
DEFAULT_FIRST_IMAGE_REL = DEFAULT_FIRST_IMAGE


def _batch_status(batch: Dict[str, Any]) -> str:
    if not batch.get("locked") or not batch.get("files"):
        return "尚未锁定目录。上传第一张图片后将锁定「工作目录」，并可用小键盘 ← → 切换。"
    name = _display_path(batch["files"][batch["index"]])
    return f"当前：{name} | {batch['index'] + 1}/{len(batch['files'])} | 目录：{batch['dir']} | ← 上一张 → 下一张 | 点击原图重新选目录"


def _resolve_batch(
    upload_path: Optional[str],
    dir_text: str,
    batch: Optional[Dict[str, Any]],
    nav: int = 0,
) -> Tuple[Dict[str, Any], Optional[str], str]:
    batch = dict(batch or _empty_batch())
    nav = int(nav or 0)

    if not batch.get("locked"):
        dir_path = _resolve_dir_path(dir_text)
        files = _list_images_in_dir(dir_path)
        if not files:
            return batch, None, f"目录中没有图片：{_display_path(dir_path)}"
        batch = {"locked": True, "dir": _display_path(dir_path), "files": files, "index": 0}
        if upload_path:
            upload_name = Path(upload_path).name
            for i, fp in enumerate(files):
                if Path(fp).name == upload_name:
                    batch["index"] = i
                    break
    elif nav != 0 and batch.get("files"):
        batch["index"] = (int(batch["index"]) + nav) % len(batch["files"])

    if not batch.get("files"):
        return batch, None, "目录中没有可用图片"
    batch["index"] = max(0, min(int(batch["index"]), len(batch["files"]) - 1))
    current = batch["files"][batch["index"]]
    return batch, current, _batch_status(batch)


def _open_image(image):
    if image is None:
        return None
    if isinstance(image, str):
        return Image.open(_resolve_image_path(image)).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def _status_prefix(runtime_info: Dict[str, Any] | None = None) -> str:
    return format_runtime_status(runtime_info or RUNTIME)


def _format_elapsed_display(
    elapsed: float | None = None,
    width: int | None = None,
    height: int | None = None,
    core_elapsed: float | None = None,
) -> str:
    if elapsed is None:
        time_part = "用时：—"
    else:
        time_part = f"用时：{elapsed:.3f} 秒"
    if core_elapsed is not None:
        time_part += f"　|　除可视化：{core_elapsed:.3f} 秒"
    if width and height:
        return f"{time_part}　|　{width}×{height}"
    return time_part


def _image_size(image) -> tuple[int | None, int | None]:
    if image is None:
        return None, None
    if isinstance(image, str):
        path = _resolve_image_path(image)
        if not path.is_file():
            return None, None
        img = cv2.imread(str(path))
        if img is None:
            return None, None
        h, w = img.shape[:2]
        return w, h
    if hasattr(image, "size"):
        w, h = image.size
        return int(w), int(h)
    arr = np.asarray(image)
    if arr.ndim >= 2:
        h, w = arr.shape[:2]
        return int(w), int(h)
    return None, None


def _format_stage_times(stage_times: Dict[str, Any] | None, elapsed: float) -> str:
    lines = []
    skip = {"总流程", "_result_core", "除可视化用时"}
    if stage_times:
        core = stage_times.get("除可视化用时")
        if core is not None:
            try:
                lines.append(f"除可视化用时：{float(core):.3f} 秒")
            except Exception:
                pass
        for key, value in stage_times.items():
            if key in skip:
                continue
            try:
                lines.append(f"{key}：{float(value):.3f} 秒")
            except Exception:
                pass
    return "\n".join(lines)


def _badge_lines(plan: Dict[str, str], kind: str) -> list[str]:
    if kind == "landmark":
        return [f"人脸/关键点 {plan['face']}"]
    if kind == "highlight":
        lines = [f"高光 {plan['highlight']}"]
        if plan["refine"] not in ("—", "关"):
            lines.append(f"精修 {plan['refine']}")
        return lines
    if kind == "region":
        return [f"区域 {plan['region']}", f"人脸/关键点 {plan['face']}"]
    return [
        f"人脸/关键点 {plan['face']}",
        f"区域 {plan['region']}",
        f"高光 {plan['highlight']}",
        f"精修 {plan['refine']}",
        f"修复 {plan['removal']}",
    ]


def _annotate_view(image_bgr: np.ndarray, plan: Dict[str, str], kind: str) -> np.ndarray:
    return annotate_resolution_badge(image_bgr, _badge_lines(plan, kind))


def _build_run_report(
    out,
    cfg: Dict[str, Any],
    runtime_info: Dict[str, Any],
    elapsed: float,
    stage_times: Dict[str, Any] | None,
    iw: int,
    ih: int,
) -> str:
    plan = describe_resolution_plan(cfg.get("pipeline", {}))
    stage_text = _format_stage_times(stage_times, elapsed)
    core_elapsed = None
    if stage_times and "除可视化用时" in stage_times:
        try:
            core_elapsed = float(stage_times["除可视化用时"])
        except Exception:
            core_elapsed = None
    parts = [
        f"状态：{out.status}",
        f"总用时：{elapsed:.3f} 秒 | 检测到人脸 {out.detection_count} 个 | 输出 {iw}×{ih}",
    ]
    if core_elapsed is not None:
        parts.append(f"除可视化用时：{core_elapsed:.3f} 秒（至去高光结果图生成完成）")
    parts.extend([
        f"运行模式：{runtime_info.get('mode', 'CPU正常模式')}",
        "",
        "分辨率方案",
        f"  人脸/关键点：{plan['face']}",
        f"  区域：{plan['region']}",
        f"  高光检测：{plan['highlight']}",
        f"  高光精修：{plan['refine']}",
        f"  去高光修复：{plan['removal']}",
        "",
        out.metrics_text,
    ])
    if stage_text:
        parts.extend(["", "耗时明细", stage_text])
    return "\n".join(parts)


def _collect_config(runtime_info: Dict[str, Any], *values) -> Dict[str, Any]:
    flat = dict(zip(_COLLECT_CONFIG_FLAT_KEYS, values))
    cfg = update_config_from_flat(deepcopy(DEFAULT_CONFIG), flat)
    flat_set = set(_COLLECT_CONFIG_FLAT_KEYS)
    if USER_SESSION_PATH.is_file():
        try:
            saved = (load_yaml(USER_SESSION_PATH).get("config") or {})
            for section, abbrev in _FLAT_SECTION_PREFIX.items():
                sec_saved = saved.get(section) or {}
                if section not in cfg or not isinstance(sec_saved, dict):
                    continue
                for key, val in sec_saved.items():
                    if f"{abbrev}__{key}" not in flat_set:
                        cfg[section][key] = val
        except Exception:
            pass
    cfg["runtime"] = runtime_info
    return cfg


def _config_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for section in ("pipeline", "face_detection", "regions", "highlight_detection", "highlight_removal"):
        if section not in cfg:
            continue
        base_keys = DEFAULT_CONFIG.get(section, {})
        snapshot[section] = {k: cfg[section][k] for k in base_keys if k in cfg.get(section, {})}
    return snapshot


def _merge_detection_sensitivity(name: str) -> Dict[str, Any]:
    hd = deepcopy(DEFAULT_CONFIG["highlight_detection"])
    if name != DEFAULT_DETECTION_SENSITIVITY and DETECTION_SENSITIVITY_PATH.is_file():
        data = load_yaml(DETECTION_SENSITIVITY_PATH)
        hd.update((data.get("presets") or {}).get(name) or {})
    return hd


def _detection_control_values(hd: Dict[str, Any]) -> List[Any]:
    return [
        hd["rgb_brightness_threshold"],
        hd["hsv_v_threshold"],
        hd["hsv_s_upper"],
        hd["lab_l_threshold"],
        hd["local_brightness_threshold"],
        hd["local_contrast_threshold"],
        hd["saturation_pixel_threshold"],
        hd["highlight_min_area"],
        hd["highlight_max_area_ratio"],
        hd["mask_dilate_radius"],
        hd["mask_erode_radius"],
        hd["mask_blur_radius"],
        hd["oil_shine_s_upper"],
        hd["adaptive_region_delta"],
        hd["adaptive_core_delta"],
        hd["adaptive_grow_iterations"],
        hd["morph_close_radius"],
        hd["soft_mask_gain"],
        hd["forehead_region_max_fraction"],
        hd["nose_tip_region_max_fraction"],
        hd["cheek_region_max_fraction"],
    ]


def apply_detection_sensitivity(sensitivity_name: str) -> List[Any]:
    name = str(sensitivity_name or DEFAULT_DETECTION_SENSITIVITY)
    if name not in DETECTION_SENSITIVITY_NAMES:
        name = DEFAULT_DETECTION_SENSITIVITY
    return _detection_control_values(_merge_detection_sensitivity(name))


def _removal_control_values(rm: Dict[str, Any]) -> List[Any]:
    return [
        rm["mode"],
        rm["brightness_suppress_strength"],
        rm["chroma_restore_strength"],
        rm["texture_preserve_strength"],
        rm["edge_protect_strength"],
        rm["inpainting_radius"],
        rm["poisson_alpha_strength"],
        rm["final_blend_alpha"],
        rm["faithful_luminance_floor"],
        rm["extreme_core_extra_l"],
        rm["max_allowed_modify_area_ratio"],
        rm["max_allowed_mean_brightness_change"],
        rm["max_allowed_local_color_delta"],
    ]


def _merge_removal_intensity(name: str) -> Dict[str, Any]:
    rm = deepcopy(DEFAULT_CONFIG["highlight_removal"])
    if name != DEFAULT_REMOVAL_INTENSITY and REMOVAL_INTENSITY_PATH.is_file():
        data = load_yaml(REMOVAL_INTENSITY_PATH)
        rm.update((data.get("presets") or {}).get(name) or {})
    return rm


def apply_removal_intensity(intensity_name: str) -> List[Any]:
    name = str(intensity_name or DEFAULT_REMOVAL_INTENSITY)
    if name not in REMOVAL_INTENSITY_NAMES:
        name = DEFAULT_REMOVAL_INTENSITY
    return _removal_control_values(_merge_removal_intensity(name))


_COLLECT_CONFIG_FLAT_KEYS = [
    "pl__process_scale",
    "pl__refine_upsampled_masks",
    "pl__enable_roi_removal",
    "fd__face_detection_confidence",
    "fd__landmark_detection_confidence",
    "fd__enable_mediapipe",
    "fd__enable_fallback_detector",
    "fd__enable_alignment",
    "fd__show_landmark_index",
    "fd__multi_face_strategy",
    "fd__manual_face_index",
    "rg__nose_bridge_expand_ratio",
    "rg__nose_tip_expand_ratio",
    "rg__forehead_height_ratio",
    "rg__cheek_expand_ratio",
    "rg__chin_expand_ratio",
    "rg__protect_expand_radius",
    "rg__eye_protect_extra_radius",
    "rg__protect_brows",
    "rg__brow_protect_shrink_radius",
    "rg__brow_protect_height_ratio",
    "rg__brow_protect_width_ratio",
    "rg__brow_vertical_shift_ratio",
    "rg__skin_mask_smooth_radius",
    "rg__face_contour_shrink_ratio",
    "rg__key_region_detection_mode",
    "hd__rgb_brightness_threshold",
    "hd__hsv_v_threshold",
    "hd__hsv_s_upper",
    "hd__lab_l_threshold",
    "hd__local_brightness_threshold",
    "hd__local_contrast_threshold",
    "hd__saturation_pixel_threshold",
    "hd__highlight_min_area",
    "hd__highlight_max_area_ratio",
    "hd__mask_dilate_radius",
    "hd__mask_erode_radius",
    "hd__mask_blur_radius",
    "hd__oil_shine_s_upper",
    "hd__adaptive_region_delta",
    "hd__adaptive_core_delta",
    "hd__adaptive_grow_iterations",
    "hd__morph_close_radius",
    "hd__soft_mask_gain",
    "hd__forehead_region_max_fraction",
    "hd__nose_tip_region_max_fraction",
    "hd__cheek_region_max_fraction",
    "rm__mode",
    "rm__brightness_suppress_strength",
    "rm__chroma_restore_strength",
    "rm__texture_preserve_strength",
    "rm__edge_protect_strength",
    "rm__inpainting_radius",
    "rm__poisson_alpha_strength",
    "rm__final_blend_alpha",
    "rm__faithful_luminance_floor",
    "rm__extreme_core_extra_l",
    "rm__max_allowed_modify_area_ratio",
    "rm__max_allowed_mean_brightness_change",
    "rm__max_allowed_local_color_delta",
]
_FLAT_SECTION_PREFIX = {
    "pipeline": "pl",
    "face_detection": "fd",
    "regions": "rg",
    "highlight_detection": "hd",
    "highlight_removal": "rm",
}


def _infer_detection_sensitivity(hd: Dict[str, Any]) -> str:
    if not DETECTION_SENSITIVITY_PATH.is_file():
        return DEFAULT_DETECTION_SENSITIVITY
    presets = (load_yaml(DETECTION_SENSITIVITY_PATH).get("presets") or {})
    for name in DETECTION_SENSITIVITY_NAMES:
        if name == DEFAULT_DETECTION_SENSITIVITY:
            continue
        preset_only = presets.get(name) or {}
        if not preset_only:
            continue
        mismatches = sum(1 for k, v in preset_only.items() if hd.get(k) != v)
        if mismatches <= 1:
            return name
    return DEFAULT_DETECTION_SENSITIVITY


def _infer_removal_intensity(rm: Dict[str, Any]) -> str:
    if not REMOVAL_INTENSITY_PATH.is_file():
        return DEFAULT_REMOVAL_INTENSITY
    presets = (load_yaml(REMOVAL_INTENSITY_PATH).get("presets") or {})
    for name in REMOVAL_INTENSITY_NAMES:
        if name == DEFAULT_REMOVAL_INTENSITY:
            continue
        preset_only = presets.get(name) or {}
        if not preset_only:
            continue
        mismatches = sum(1 for k, v in preset_only.items() if rm.get(k) != v)
        if mismatches == 0:
            return name
    return DEFAULT_REMOVAL_INTENSITY


def _normalize_image_dir_text(dir_text: str | Path | None) -> str:
    return _display_path(_resolve_dir_path(dir_text))


def _batch_snapshot(batch: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    batch = dict(batch or {})
    if not batch.get("locked") or not batch.get("files"):
        return {}
    index = max(0, min(int(batch.get("index", 0)), len(batch["files"]) - 1))
    return {
        "index": index,
        "dir": str(batch.get("dir") or ""),
        "current_file": _display_path(batch["files"][index]),
    }


def _restore_batch(saved_batch: Optional[Dict[str, Any]], image_dir: str | Path | None) -> Dict[str, Any]:
    saved_batch = dict(saved_batch or {})
    dir_text = saved_batch.get("dir") or image_dir
    dir_text = str(dir_text or "").strip() or DEFAULT_IMAGE_DIR_REL
    dir_path = _resolve_dir_path(dir_text)
    files = _list_images_in_dir(dir_path)
    if not files:
        return _empty_batch()
    batch: Dict[str, Any] = {
        "locked": True,
        "dir": _display_path(dir_path),
        "files": files,
        "index": 0,
    }
    if saved_batch.get("index") is not None:
        batch["index"] = max(0, min(int(saved_batch["index"]), len(files) - 1))
    current_file = str(saved_batch.get("current_file") or "").strip()
    if current_file:
        target = _resolve_image_path(current_file)
        for i, fp in enumerate(files):
            fp_path = _resolve_image_path(fp)
            if fp_path == target or fp_path.name == target.name:
                batch["index"] = i
                break
    return batch


def _input_values_from_config(cfg: Dict[str, Any], runtime_mode: str) -> List[Any]:
    fd = cfg["face_detection"]
    rg = cfg["regions"]
    hd = cfg["highlight_detection"]
    rm = cfg["highlight_removal"]
    pl = cfg.get("pipeline", {})
    return [
        runtime_mode,
        pl.get("process_scale", 0.5),
        pl.get("refine_upsampled_masks", False),
        pl.get("enable_roi_removal", True),
        fd["face_detection_confidence"],
        fd["landmark_detection_confidence"],
        fd["enable_mediapipe"],
        fd["enable_fallback_detector"],
        fd["enable_alignment"],
        fd["show_landmark_index"],
        fd["multi_face_strategy"],
        fd["manual_face_index"],
        rg["nose_bridge_expand_ratio"],
        rg["nose_tip_expand_ratio"],
        rg["forehead_height_ratio"],
        rg["cheek_expand_ratio"],
        rg["chin_expand_ratio"],
        rg["protect_expand_radius"],
        rg.get("eye_protect_extra_radius", 0.5),
        rg.get("protect_brows", False),
        rg.get("brow_protect_shrink_radius", 2),
        rg.get("brow_protect_height_ratio", 1.30),
        rg.get("brow_protect_width_ratio", 1.20),
        rg.get("brow_vertical_shift_ratio", 0.12),
        rg["skin_mask_smooth_radius"],
        rg["face_contour_shrink_ratio"],
        rg.get("key_region_detection_mode", False),
        hd["rgb_brightness_threshold"],
        hd["hsv_v_threshold"],
        hd["hsv_s_upper"],
        hd["lab_l_threshold"],
        hd["local_brightness_threshold"],
        hd["local_contrast_threshold"],
        hd["saturation_pixel_threshold"],
        hd["highlight_min_area"],
        hd["highlight_max_area_ratio"],
        hd["mask_dilate_radius"],
        hd["mask_erode_radius"],
        hd["mask_blur_radius"],
        hd["oil_shine_s_upper"],
        hd["adaptive_region_delta"],
        hd["adaptive_core_delta"],
        hd["adaptive_grow_iterations"],
        hd["morph_close_radius"],
        hd["soft_mask_gain"],
        hd["forehead_region_max_fraction"],
        hd["nose_tip_region_max_fraction"],
        hd["cheek_region_max_fraction"],
        rm["mode"],
        rm["brightness_suppress_strength"],
        rm["chroma_restore_strength"],
        rm["texture_preserve_strength"],
        rm["edge_protect_strength"],
        rm["inpainting_radius"],
        rm["poisson_alpha_strength"],
        rm["final_blend_alpha"],
        rm["faithful_luminance_floor"],
        rm["extreme_core_extra_l"],
        rm["max_allowed_modify_area_ratio"],
        rm["max_allowed_mean_brightness_change"],
        rm["max_allowed_local_color_delta"],
    ]


def _load_startup_state() -> Tuple[Dict[str, Any], str, str, str, str, Dict[str, Any]]:
    cfg = deepcopy(DEFAULT_CONFIG)
    runtime_mode = DEFAULT_RUNTIME_MODE
    image_dir = DEFAULT_IMAGE_DIR_REL
    detection_sensitivity = DEFAULT_DETECTION_SENSITIVITY
    removal_intensity = DEFAULT_REMOVAL_INTENSITY
    saved_batch: Dict[str, Any] = {}
    if not USER_SESSION_PATH.is_file():
        return cfg, runtime_mode, image_dir, detection_sensitivity, removal_intensity, saved_batch
    try:
        session = load_yaml(USER_SESSION_PATH)
    except Exception:
        return cfg, runtime_mode, image_dir, detection_sensitivity, removal_intensity, saved_batch
    saved_cfg = session.get("config") or {}
    for section, values in saved_cfg.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
    runtime_mode = normalize_runtime_mode(session.get("runtime_mode"))
    if runtime_mode not in RUNTIME_MODES:
        runtime_mode = DEFAULT_RUNTIME_MODE
    image_dir = _normalize_image_dir_text(session.get("image_dir") or image_dir)
    detection_sensitivity = str(session.get("detection_sensitivity") or _infer_detection_sensitivity(cfg["highlight_detection"]))
    if detection_sensitivity not in DETECTION_SENSITIVITY_NAMES:
        detection_sensitivity = _infer_detection_sensitivity(cfg["highlight_detection"])
    removal_intensity = str(session.get("removal_intensity") or _infer_removal_intensity(cfg["highlight_removal"]))
    if removal_intensity not in REMOVAL_INTENSITY_NAMES:
        removal_intensity = _infer_removal_intensity(cfg["highlight_removal"])
    saved_batch = dict(session.get("batch") or {})
    if saved_batch.get("dir"):
        saved_batch["dir"] = _normalize_stored_path(str(saved_batch["dir"]))
    if saved_batch.get("current_file"):
        saved_batch["current_file"] = _normalize_stored_path(str(saved_batch["current_file"]))
    return cfg, runtime_mode, image_dir, detection_sensitivity, removal_intensity, saved_batch


def _persist_user_session(
    runtime_mode: str,
    image_dir: str,
    detection_sensitivity: str,
    removal_intensity: str,
    batch: Optional[Dict[str, Any]],
    *values,
) -> None:
    runtime_info = apply_runtime_mode(normalize_runtime_mode(runtime_mode))
    cfg = _collect_config(runtime_info, *values)
    payload = {
        "runtime_mode": str(runtime_mode or DEFAULT_RUNTIME_MODE),
        "image_dir": _normalize_image_dir_text(image_dir),
        "detection_sensitivity": str(detection_sensitivity or DEFAULT_DETECTION_SENSITIVITY),
        "removal_intensity": str(removal_intensity or DEFAULT_REMOVAL_INTENSITY),
        "config": _config_snapshot(cfg),
        "batch": _batch_snapshot(batch),
    }
    save_yaml(payload, USER_SESSION_PATH)


def _persist_user_session_ui(
    image_dir: str,
    detection_sensitivity: str,
    removal_intensity: str,
    batch: Optional[Dict[str, Any]],
    runtime_mode: str,
    *values,
) -> None:
    _persist_user_session(runtime_mode, image_dir, detection_sensitivity, removal_intensity, batch, *values)


def _save_outputs(out, cfg):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 文件名使用英文，避免 Windows + OpenCV 中文路径保存失败
    result_path = OUTPUT_DIR / f"result_{stamp}.png"
    mask_path = OUTPUT_DIR / f"highlight_mask_{stamp}.png"
    params_path = OUTPUT_DIR / f"params_{stamp}.yaml"

    result_file = write_image(result_path, out.result_bgr)

    if out.highlight_mask.ndim == 2:
        mask_bgr = cv2.cvtColor(out.highlight_mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bgr = out.highlight_mask
    mask_file = write_image(mask_path, mask_bgr)

    save_yaml(cfg, params_path)

    if not _resolve_image_path(result_file).exists():
        raise FileNotFoundError(f"去高光结果保存失败：{result_file}")
    if not _resolve_image_path(mask_file).exists():
        raise FileNotFoundError(f"高光 Mask 保存失败：{mask_file}")
    if not _resolve_image_path(params_path).exists():
        raise FileNotFoundError(f"参数配置保存失败：{params_path}")

    return (
        _normalize_stored_path(result_file),
        _normalize_stored_path(mask_file),
        _normalize_stored_path(params_path),
    )


def process_ui(image, runtime_mode, *values):
    runtime_info = apply_runtime_mode(normalize_runtime_mode(runtime_mode))
    iw, ih = _image_size(image)
    image = _open_image(image)
    if image is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        msg = "请先上传证件照。\n" + _status_prefix(runtime_info)
        return [
            blank, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank,
            msg, "## 输出结果", None, None, None, _format_elapsed_display(None, iw, ih),
        ]
    try:
        cfg = _collect_config(runtime_info, *values)
        image_bgr = pil_to_bgr(image)
        ih, iw = image_bgr.shape[:2]
        t0 = time.perf_counter()
        out = process_image(image_bgr, cfg)
        elapsed = time.perf_counter() - t0
        core_elapsed = float((out.stage_times or {}).get("除可视化用时", elapsed))
        plan = describe_resolution_plan(cfg.get("pipeline", {}))
        heading = f"## 输出结果（用时 {elapsed:.3f} 秒）"
        elapsed_display = _format_elapsed_display(elapsed, iw, ih, core_elapsed)
        if not out.success:
            report = _build_run_report(out, cfg, runtime_info, elapsed, getattr(out, "stage_times", {}), iw, ih)
            rgb = bgr_to_rgb(image_bgr)
            blank = np.zeros_like(rgb)
            return [
                rgb, blank, rgb, rgb, blank, blank, blank, blank, blank, blank, blank,
                report, heading, None, None, None, elapsed_display,
            ]
        result_file, mask_file, params_file = _save_outputs(out, cfg)
        report = _build_run_report(out, cfg, runtime_info, elapsed, getattr(out, "stage_times", {}), iw, ih)
        highlight_bgr = out.highlight_mask if out.highlight_mask.ndim == 3 else cv2.cvtColor(out.highlight_mask, cv2.COLOR_GRAY2BGR)
        return [
            bgr_to_rgb(_annotate_view(out.result_bgr, plan, "full")),
            bgr_to_rgb(_annotate_view(out.face_region_view_bgr, plan, "region")),
            bgr_to_rgb(_annotate_view(highlight_bgr, plan, "highlight")),
            bgr_to_rgb(_annotate_view(out.landmark_view_bgr, plan, "landmark")),
            bgr_to_rgb(out.skin_mask_bgr),
            bgr_to_rgb(out.protect_mask_bgr),
            bgr_to_rgb(out.nose_bridge_bgr),
            bgr_to_rgb(out.forehead_bgr),
            bgr_to_rgb(out.left_cheek_bgr),
            bgr_to_rgb(out.right_cheek_bgr),
            bgr_to_rgb(_annotate_view(out.diff_bgr, plan, "full")),
            report,
            heading,
            _gradio_serve_path(result_file),
            _gradio_serve_path(mask_file),
            _gradio_serve_path(params_file),
            elapsed_display,
        ]
    except Exception as exc:
        rgb = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        blank = np.zeros_like(rgb)
        ih, iw = rgb.shape[:2]
        msg = f"处理失败：{exc}\n未检测到可靠人脸，请更换图片或调低检测阈值。\n{_status_prefix(runtime_info)}"
        return [
            rgb, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank,
            msg, "## 输出结果", None, None, None, _format_elapsed_display(None, iw, ih),
        ]


def run_batch_ui(
    upload_path,
    dir_text,
    batch,
    nav,
    runtime_mode,
    detection_sensitivity,
    removal_intensity,
    *values,
    update_input_image: bool = False,
):
    batch, current_path, status = _resolve_batch(upload_path, dir_text, batch, nav)
    input_update = _gradio_serve_path(current_path) if update_input_image and current_path else gr.skip()
    if current_path is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        iw, ih = _image_size(upload_path)
        return [
            blank, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank,
            status, "## 输出结果", None, None, None, _format_elapsed_display(None, iw, ih),
            input_update, status, batch,
        ]
    outputs = process_ui(current_path, runtime_mode, *values)
    try:
        _persist_user_session(
            runtime_mode,
            dir_text,
            detection_sensitivity,
            removal_intensity,
            batch,
            *values,
        )
    except Exception:
        pass
    outputs.extend([input_update, status, batch])
    return outputs


def on_page_load(
    upload_path,
    dir_text,
    batch,
    detection_sensitivity,
    removal_intensity,
    runtime_mode,
    *values,
):
    """打开/刷新页面：从 user_session.yaml 恢复全部参数、预设档位与当前图片索引。"""
    cfg, rt, img_dir, det_sens, rem_int, saved_batch = _load_startup_state()
    restored = _input_values_from_config(cfg, rt)
    batch = _restore_batch(saved_batch, img_dir)
    if batch.get("locked") and batch.get("files"):
        current_path = batch["files"][batch["index"]]
    else:
        current_path = _first_image_in_dir(img_dir)
        batch, current_path, _ = _resolve_batch(current_path, img_dir, batch, 0)
    if not current_path:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        msg = f"目录中没有图片：{img_dir or DEFAULT_IMAGE_DIR_REL}"
        batch_out = [
            blank, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank,
            msg, "## 输出结果", None, None, None, _format_elapsed_display(None),
            gr.skip(), msg, batch,
        ]
    else:
        batch_out = run_batch_ui(
            current_path,
            img_dir,
            batch,
            0,
            restored[0],
            det_sens,
            rem_int,
            *restored[1:],
            update_input_image=True,
        )
    return [img_dir, det_sens, rem_int, batch] + restored + batch_out


def reset_batch_ui(batch):
    batch = _empty_batch()
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    return (
        blank, blank, blank, blank, blank, blank, blank, blank, blank, blank, blank,
        "已重置。请修改工作目录或重新上传第一张图片。",
        "## 输出结果",
        None,
        None,
        None,
        _format_elapsed_display(None),
        None,
        _batch_status(batch),
        batch,
    )


NAV_KEYBOARD_JS = """
() => {
  if (window.__faceHighlightNavBound) return;
  window.__faceHighlightNavBound = true;
  document.addEventListener('keydown', (event) => {
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || event.target?.isContentEditable) return;
    let btn = null;
    if (event.key === 'ArrowRight' || event.code === 'Numpad6') btn = document.getElementById('btn_next_image');
    if (event.key === 'ArrowLeft' || event.code === 'Numpad4') btn = document.getElementById('btn_prev_image');
    if (btn) { btn.click(); event.preventDefault(); }
  });
}
"""


def save_best_ui(
    image_dir: str,
    detection_sensitivity: str,
    removal_intensity: str,
    batch: Optional[Dict[str, Any]],
    runtime_mode: str,
    *values,
):
    runtime_info = apply_runtime_mode(normalize_runtime_mode(runtime_mode))
    cfg = _collect_config(runtime_info, *values)
    path = OUTPUT_DIR / "best_params.yaml"
    save_yaml(cfg, path)
    try:
        _persist_user_session(runtime_mode, image_dir, detection_sensitivity, removal_intensity, batch, *values)
    except Exception:
        pass
    return f"已保存当前最佳参数：{path}\n参数记忆已同步。\n{_status_prefix(runtime_info)}", str(path)


def reset_values(image_dir: str):
    vals = _input_values_from_config(DEFAULT_CONFIG, DEFAULT_RUNTIME_MODE)
    dir_norm = _normalize_image_dir_text(image_dir)
    batch = _restore_batch({"index": 0, "dir": dir_norm}, dir_norm)
    try:
        _persist_user_session(
            DEFAULT_RUNTIME_MODE,
            dir_norm,
            DEFAULT_DETECTION_SENSITIVITY,
            DEFAULT_REMOVAL_INTENSITY,
            batch,
            *vals[1:],
        )
    except Exception:
        pass
    return vals + [DEFAULT_DETECTION_SENSITIVITY, DEFAULT_REMOVAL_INTENSITY]


def build_app():
    startup_cfg, startup_runtime, startup_image_dir, startup_det_sens, startup_rem_int, startup_saved_batch = _load_startup_state()
    startup_batch = _restore_batch(startup_saved_batch, startup_image_dir)
    fd = startup_cfg["face_detection"]
    rg = startup_cfg["regions"]
    hd = startup_cfg["highlight_detection"]
    rm = startup_cfg["highlight_removal"]
    pl = startup_cfg.get("pipeline", {})
    app_css = """
    .elapsed-time, .elapsed-time p {
        color: #d32f2f !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        margin: 0.15rem 0 0.5rem !important;
    }
    """
    with gr.Blocks(title="公安证件照人脸面部高光去除系统", css=app_css) as demo:
        batch_state = gr.State(startup_batch)
        gr.Markdown("# 公安证件照人脸面部高光去除系统\n基于人脸检测、关键点区域追踪、皮肤区域约束和真实性审计。CPU-only：默认 CPU 狂暴（全核 + 高优先级），GPU 已禁用。引擎切换 / Python vs ONNX 对比请用精简工作室页：`python app_studio.py`（端口 7861）。")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="原图：上传第一张后锁定目录；点击原图可重新选目录",
                    type="filepath",
                    sources=["upload"],
                )
                image_dir = gr.Textbox(
                    label="工作目录（相对路径；默认第一张 ../data/1.png）",
                    value=startup_image_dir,
                )
                batch_status = gr.Textbox(
                    label="目录浏览",
                    value=_batch_status(startup_batch) if startup_batch.get("locked") else (
                        _batch_status({
                            "locked": True,
                            "dir": DEFAULT_IMAGE_DIR_REL,
                            "files": _list_images_in_dir(DEFAULT_IMAGE_DIR_REL),
                            "index": 0,
                        }) if DEFAULT_FIRST_IMAGE else _batch_status(_empty_batch())
                    ),
                    lines=2,
                    interactive=False,
                )
                with gr.Row():
                    prev_btn = gr.Button("← 上一张", elem_id="btn_prev_image")
                    next_btn = gr.Button("下一张 →", elem_id="btn_next_image")
            with gr.Column():
                runtime_mode = gr.Dropdown(RUNTIME_MODES, value=startup_runtime, label="运行模式：CPU狂暴 / CPU正常 / CPU单线程（GPU 已禁用）")
                process_scale = gr.Dropdown(
                    choices=[
                        ("1.0 原图", 1.0),
                        ("0.5 二分之一", 0.5),
                        ("0.25 四分之一", 0.25),
                        ("折中（人脸1/4 + 高光1/2）", "compromise"),
                    ],
                    value=pl.get("process_scale", 0.5),
                    label="处理压缩比例：人脸/区域用该比例；折中模式为人脸1/4 + 高光1/2",
                )
                enable_highlight_refine = gr.Checkbox(
                    value=pl.get("refine_upsampled_masks", False),
                    label="启用高光区域精修（非折中：原图高光；折中：开=原图高光 / 关=仅1/2高光上采样）",
                )
                enable_roi = gr.Checkbox(value=pl.get("enable_roi_removal", True), label="启用 ROI 局部修复（仅在高光区域附近做去高光）")
                with gr.Row():
                    detection_sensitivity = gr.Dropdown(
                        choices=DETECTION_SENSITIVITY_NAMES,
                        value=startup_det_sens,
                        label="检测敏感度（灵敏=更低阈值，易检出弱高光）",
                    )
                    removal_intensity = gr.Dropdown(
                        choices=REMOVAL_INTENSITY_NAMES,
                        value=startup_rem_int,
                        label="去高光强度（正常=默认；强力=压得更狠）",
                    )
                elapsed_display = gr.Markdown(_format_elapsed_display(None), elem_classes=["elapsed-time"])
                key_region_mode = gr.Checkbox(
                    value=rg.get("key_region_detection_mode", False),
                    label="切换回关键区域检测模式（勾选=多色分区；默认关闭=仅保护眉眼嘴）",
                )
                original_note = gr.Textbox(label="运行状态", value=_status_prefix(RUNTIME), lines=12, interactive=False)
        with gr.Row():
            run_btn = gr.Button("开始处理", variant="primary")
            reset_btn = gr.Button("重置按钮：恢复默认参数")
            save_btn = gr.Button("保存按钮：保存当前最佳参数")

        with gr.Accordion("人脸检测参数", open=False):
            fd_conf = gr.Slider(0.10, 0.95, value=fd["face_detection_confidence"], step=0.01, label="人脸检测置信度阈值")
            lm_conf = gr.Slider(0.10, 0.95, value=fd["landmark_detection_confidence"], step=0.01, label="关键点检测置信度阈值")
            enable_mp = gr.Checkbox(value=fd["enable_mediapipe"], label="是否启用 MediaPipe")
            enable_fb = gr.Checkbox(value=fd["enable_fallback_detector"], label="是否启用 fallback 检测器")
            enable_align = gr.Checkbox(value=fd["enable_alignment"], label="是否启用人脸轻量对齐")
            show_index = gr.Checkbox(value=fd["show_landmark_index"], label="是否显示关键点编号")
            multi_strategy = gr.Dropdown(["最大人脸", "置信度最高", "手动编号"], value=fd["multi_face_strategy"], label="多人脸选择方式")
            manual_idx = gr.Slider(0, 10, value=fd["manual_face_index"], step=1, label="手动选择人脸编号")

        with gr.Accordion("区域追踪参数", open=False):
            nose_bridge_expand = gr.Slider(0.04, 0.30, value=rg["nose_bridge_expand_ratio"], step=0.01, label="鼻梁区域扩张比例")
            nose_tip_expand = gr.Slider(0.30, 1.30, value=rg["nose_tip_expand_ratio"], step=0.01, label="鼻尖区域扩张比例")
            forehead_ratio = gr.Slider(
                0.70,
                2.00,
                value=rg["forehead_height_ratio"],
                step=0.01,
                label="额头区域高度比例（脸廓弧线；1.0≈发际线，>1.0再向上扩）",
            )
            cheek_expand = gr.Slider(0.70, 1.50, value=rg["cheek_expand_ratio"], step=0.01, label="左右脸颊区域扩张比例")
            chin_expand = gr.Slider(0.70, 1.50, value=rg["chin_expand_ratio"], step=0.01, label="下巴区域扩张比例")
            protect_radius = gr.Slider(1, 20, value=rg["protect_expand_radius"], step=1, label="五官保护区扩张半径")
            eye_protect_extra = gr.Slider(0, 4, value=rg.get("eye_protect_extra_radius", 0.5), step=0.5, label="眼睛保护额外扩张（0.5≈不额外膨胀；1.0=+1px）")
            protect_brows = gr.Checkbox(
                value=rg.get("protect_brows", False),
                label="眉毛红色保护（仅细线眉发受保护，眉弓处仍可检测去高光）",
            )
            brow_protect_shrink = gr.Slider(
                0, 12,
                value=rg.get("brow_protect_shrink_radius", 2),
                step=1,
                label="眉毛保护收缩（越大范围越小，与高度比例配合）",
            )
            brow_protect_height = gr.Slider(
                0.50, 2.50,
                value=rg.get("brow_protect_height_ratio", 1.30),
                step=0.05,
                label="眉毛区域高度比例（1.0=默认；越大眉带越厚）",
            )
            brow_protect_width = gr.Slider(
                0.50, 2.00,
                value=rg.get("brow_protect_width_ratio", 1.20),
                step=0.05,
                label="眉毛区域宽度比例（1.0=默认；越大左右越宽）",
            )
            brow_vertical_shift = gr.Slider(
                0.00, 0.40,
                value=rg.get("brow_vertical_shift_ratio", 0.12),
                step=0.01,
                label="眉毛整体上移（相对两眼间距；越大越靠上，默认 0.12）",
            )
            skin_smooth = gr.Slider(0, 15, value=rg["skin_mask_smooth_radius"], step=1, label="皮肤 mask 平滑半径")
            face_shrink = gr.Slider(0.00, 0.12, value=rg["face_contour_shrink_ratio"], step=0.005, label="面部轮廓 mask 收缩比例")

        with gr.Accordion("高光检测参数", open=False):
            rgb_thr = gr.Slider(150, 255, value=hd["rgb_brightness_threshold"], step=1, label="RGB 亮度阈值")
            hsv_v_thr = gr.Slider(150, 255, value=hd["hsv_v_threshold"], step=1, label="HSV V 通道阈值")
            hsv_s_upper = gr.Slider(20, 180, value=hd["hsv_s_upper"], step=1, label="HSV S 通道上限")
            lab_l_thr = gr.Slider(150, 255, value=hd["lab_l_threshold"], step=1, label="Lab L 通道阈值")
            local_b_thr = gr.Slider(1, 60, value=hd["local_brightness_threshold"], step=1, label="局部亮度异常阈值")
            local_c_thr = gr.Slider(0.01, 0.30, value=hd["local_contrast_threshold"], step=0.005, label="局部对比阈值")
            sat_thr = gr.Slider(200, 255, value=hd["saturation_pixel_threshold"], step=1, label="饱和像素阈值")
            min_area = gr.Slider(1, 500, value=hd["highlight_min_area"], step=1, label="高光 mask 最小面积")
            max_area_ratio = gr.Slider(0.01, 0.30, value=hd["highlight_max_area_ratio"], step=0.005, label="高光 mask 最大面积比例")
            dilate_r = gr.Slider(0, 15, value=hd["mask_dilate_radius"], step=1, label="mask 膨胀半径")
            erode_r = gr.Slider(0, 15, value=hd["mask_erode_radius"], step=1, label="mask 腐蚀半径")
            blur_r = gr.Slider(0, 25, value=hd["mask_blur_radius"], step=1, label="mask 模糊半径")
            oil_s_upper = gr.Slider(60, 220, value=hd["oil_shine_s_upper"], step=1, label="暖色油光饱和度上限")
            adaptive_delta = gr.Slider(1.0, 20.0, value=hd["adaptive_region_delta"], step=0.1, label="区域自适应高光晕阈值")
            adaptive_core_delta = gr.Slider(2.0, 30.0, value=hd["adaptive_core_delta"], step=0.1, label="区域自适应高光核心阈值")
            adaptive_grow = gr.Slider(0, 30, value=hd["adaptive_grow_iterations"], step=1, label="高光核心向油光晕扩展次数")
            close_r = gr.Slider(0, 10, value=hd["morph_close_radius"], step=1, label="mask 闭运算半径")
            soft_gain = gr.Slider(0.5, 3.0, value=hd["soft_mask_gain"], step=0.05, label="soft mask 强度增益")
            forehead_frac = gr.Slider(0.05, 0.60, value=hd["forehead_region_max_fraction"], step=0.01, label="额头区域最大高光比例")
            nose_tip_frac = gr.Slider(0.05, 0.85, value=hd["nose_tip_region_max_fraction"], step=0.01, label="鼻尖区域最大高光比例")
            cheek_frac = gr.Slider(0.03, 0.45, value=hd["cheek_region_max_fraction"], step=0.01, label="脸颊区域最大高光比例")

        with gr.Accordion("去高光参数", open=False):
            mode = gr.Dropdown(["保真", "强修复", "混合"], value=rm["mode"], label="去高光模式：保真 / 强修复 / 混合")
            bright_strength = gr.Slider(0.0, 1.0, value=rm["brightness_suppress_strength"], step=0.01, label="亮度压制强度")
            chroma_strength = gr.Slider(0.0, 1.0, value=rm["chroma_restore_strength"], step=0.01, label="色度恢复强度")
            texture_strength = gr.Slider(0.0, 1.0, value=rm["texture_preserve_strength"], step=0.01, label="纹理保留强度")
            edge_strength = gr.Slider(0.0, 1.0, value=rm["edge_protect_strength"], step=0.01, label="边缘保护强度")
            inpaint_radius = gr.Slider(1, 9, value=rm["inpainting_radius"], step=1, label="inpainting 半径")
            poisson_alpha = gr.Slider(0.0, 1.0, value=rm["poisson_alpha_strength"], step=0.01, label="Poisson / alpha 融合强度")
            final_alpha = gr.Slider(0.0, 1.0, value=rm["final_blend_alpha"], step=0.01, label="最终融合 alpha")
            faithful_l_floor = gr.Slider(
                0.0, 1.0,
                value=rm.get("faithful_luminance_floor", 0.76),
                step=0.01,
                label="保真亮度下限（越低越敢压白点；建议 0.60～0.65 去残留高光）",
            )
            extreme_core_l = gr.Slider(
                0, 30,
                value=rm.get("extreme_core_extra_l", 12),
                step=1,
                label="极端核心 L 超出量（L≥Lab L阈值+此值触发二次 inpaint；建议 6～8）",
            )
            max_mod_area = gr.Slider(0.01, 0.30, value=rm["max_allowed_modify_area_ratio"], step=0.005, label="最大允许修改面积比例")
            max_mean_l = gr.Slider(1, 60, value=rm["max_allowed_mean_brightness_change"], step=1, label="最大允许平均亮度变化")
            max_delta = gr.Slider(1, 60, value=rm["max_allowed_local_color_delta"], step=1, label="最大允许局部色差变化")

        inputs = [
            runtime_mode,
            process_scale, enable_highlight_refine, enable_roi,
            fd_conf, lm_conf, enable_mp, enable_fb, enable_align, show_index, multi_strategy, manual_idx,
            nose_bridge_expand, nose_tip_expand, forehead_ratio, cheek_expand, chin_expand, protect_radius, eye_protect_extra, protect_brows, brow_protect_shrink, brow_protect_height, brow_protect_width, brow_vertical_shift, skin_smooth, face_shrink, key_region_mode,
            rgb_thr, hsv_v_thr, hsv_s_upper, lab_l_thr, local_b_thr, local_c_thr, sat_thr, min_area, max_area_ratio, dilate_r, erode_r, blur_r,
            oil_s_upper, adaptive_delta, adaptive_core_delta, adaptive_grow, close_r, soft_gain, forehead_frac, nose_tip_frac, cheek_frac,
            mode, bright_strength, chroma_strength, texture_strength, edge_strength, inpaint_radius, poisson_alpha, final_alpha, faithful_l_floor, extreme_core_l, max_mod_area, max_mean_l, max_delta,
        ]

        detection_param_outputs = [
            rgb_thr, hsv_v_thr, hsv_s_upper, lab_l_thr, local_b_thr, local_c_thr, sat_thr,
            min_area, max_area_ratio, dilate_r, erode_r, blur_r, oil_s_upper,
            adaptive_delta, adaptive_core_delta, adaptive_grow, close_r, soft_gain,
            forehead_frac, nose_tip_frac, cheek_frac,
        ]
        removal_param_outputs = [
            mode, bright_strength, chroma_strength, texture_strength, edge_strength,
            inpaint_radius, poisson_alpha, final_alpha, faithful_l_floor, extreme_core_l,
            max_mod_area, max_mean_l, max_delta,
        ]

        output_heading = gr.Markdown("## 输出结果")
        with gr.Row():
            result_img = gr.Image(label="处理结果：去高光结果")
            region_img = gr.Image(label="人脸区域：人脸区域检测结果")
        with gr.Row():
            highlight_img = gr.Image(label="高光区域：高光检测区域")
            landmark_img = gr.Image(label="人脸关键点：人脸关键点可视化")
        with gr.Row():
            skin_img = gr.Image(label="皮肤区域：皮肤区域 Mask")
            protect_img = gr.Image(label="保护区域：五官保护区域")
        with gr.Row():
            nose_img = gr.Image(label="鼻梁区域：鼻梁高光候选区")
            forehead_img = gr.Image(label="额头区域：额头高光候选区")
        with gr.Row():
            left_cheek_img = gr.Image(label="左脸颊区域：左脸颊高光候选区")
            right_cheek_img = gr.Image(label="右脸颊区域：右脸颊高光候选区")
        with gr.Row():
            diff_img = gr.Image(label="差分图：修改差分图")
        report_text = gr.Textbox(label="运行报告", lines=18, interactive=False)
        with gr.Row():
            result_file = gr.File(label="下载按钮：下载去高光结果")
            mask_file = gr.File(label="下载 mask：下载高光 Mask")
            params_file = gr.File(label="下载参数：下载当前参数配置")

        batch_outputs = [
            result_img, region_img, highlight_img, landmark_img, skin_img, protect_img,
            nose_img, forehead_img, left_cheek_img, right_cheek_img, diff_img,
            report_text, output_heading, result_file, mask_file, params_file,
            elapsed_display,
            input_image, batch_status, batch_state,
        ]
        batch_inputs = [input_image, image_dir, batch_state, detection_sensitivity, removal_intensity] + inputs

        def _run_nav(nav, update_input_image=False):
            def _fn(upload_path, dir_text, batch, detection_sensitivity, removal_intensity, runtime_mode, *values):
                return run_batch_ui(
                    upload_path,
                    dir_text,
                    batch,
                    nav,
                    runtime_mode,
                    detection_sensitivity,
                    removal_intensity,
                    *values,
                    update_input_image=update_input_image,
                )

            return _fn

        session_persist_inputs = [image_dir, detection_sensitivity, removal_intensity, batch_state] + inputs
        page_load_outputs = session_persist_inputs + batch_outputs

        run_btn.click(_run_nav(0, False), inputs=batch_inputs, outputs=batch_outputs)
        input_image.upload(_run_nav(0, False), inputs=batch_inputs, outputs=batch_outputs)
        prev_btn.click(_run_nav(-1, True), inputs=batch_inputs, outputs=batch_outputs)
        next_btn.click(_run_nav(1, True), inputs=batch_inputs, outputs=batch_outputs)
        auto_run = _run_nav(0, False)
        for ctrl in (runtime_mode, process_scale, enable_highlight_refine, enable_roi):
            ctrl.change(auto_run, inputs=batch_inputs, outputs=batch_outputs)
        detection_sensitivity.change(
            apply_detection_sensitivity,
            inputs=[detection_sensitivity],
            outputs=detection_param_outputs,
        ).then(auto_run, inputs=batch_inputs, outputs=batch_outputs).then(
            _persist_user_session_ui,
            inputs=session_persist_inputs,
            outputs=[],
            show_progress="hidden",
        )
        removal_intensity.change(
            apply_removal_intensity,
            inputs=[removal_intensity],
            outputs=removal_param_outputs,
        ).then(auto_run, inputs=batch_inputs, outputs=batch_outputs).then(
            _persist_user_session_ui,
            inputs=session_persist_inputs,
            outputs=[],
            show_progress="hidden",
        )
        for ctrl in [image_dir] + inputs:
            ctrl.change(
                _persist_user_session_ui,
                inputs=session_persist_inputs,
                outputs=[],
                show_progress="hidden",
            )
        input_image.select(reset_batch_ui, inputs=[batch_state], outputs=batch_outputs)
        save_btn.click(save_best_ui, inputs=session_persist_inputs, outputs=[report_text, params_file])
        reset_btn.click(reset_values, inputs=[image_dir], outputs=inputs + [detection_sensitivity, removal_intensity])
        demo.load(on_page_load, inputs=batch_inputs, outputs=page_load_outputs)
        demo.load(None, None, None, js=NAV_KEYBOARD_JS)
    return demo


if __name__ == "__main__":
    preload_fd = deepcopy(DEFAULT_CONFIG["face_detection"])
    preload_fd["_runtime"] = RUNTIME
    model_path = (ROOT / "models" / "face_landmarker.task").resolve()
    preload_fd["mediapipe_task_model_path"] = str(model_path)
    if preload_face_landmarker(preload_fd):
        print("FaceLandmarker preloaded.")
    else:
        print(f"FaceLandmarker preload failed: {model_path}")
    data_dir = _resolve_dir_path(DEFAULT_IMAGE_DIR_REL)
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(data_dir.resolve()), str(OUTPUT_DIR.resolve()), str(ROOT.resolve())],
    )
