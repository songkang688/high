from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import time

from .face_detect import DetectOutput, FaceResult, detect_face, scale_face_to_shape
from .face_landmarks import bbox_from_landmarks, get_geometry
from .face_regions import RegionMasks, build_face_regions
from .highlight_detect import HighlightOutput, detect_highlight
from .quality_metrics import compute_metrics, metrics_to_text
from .remove_highlight import RemovalOutput, remove_highlight
from .utils import (
    clamp_process_scale,
    mask_to_uint8,
    recursive_update,
    resize_for_process,
    resolve_process_scales,
    scale_landmarks,
    scaled_highlight_params,
    upsample_mask_hard,
    upsample_mask_soft,
)
from .visualization import draw_highlight_view, draw_landmark_view, draw_region_view, mask_preview


@dataclass
class PipelineOutput:
    success: bool
    status: str
    warnings: List[str]
    result_bgr: np.ndarray
    highlight_mask: np.ndarray
    soft_mask: np.ndarray
    diff_bgr: np.ndarray
    landmark_view_bgr: np.ndarray
    face_region_view_bgr: np.ndarray
    skin_mask_bgr: np.ndarray
    protect_mask_bgr: np.ndarray
    nose_bridge_bgr: np.ndarray
    forehead_bgr: np.ndarray
    left_cheek_bgr: np.ndarray
    right_cheek_bgr: np.ndarray
    metrics: Dict[str, Any]
    metrics_text: str
    params: Dict[str, Any]
    regions: Optional[RegionMasks] = None
    face: Optional[FaceResult] = None
    detection_count: int = 0
    iteration_note: str = ""
    stage_times: Dict[str, float] = field(default_factory=dict)


def _merge_removal_params(config: Dict[str, Any]) -> Dict[str, Any]:
    p = deepcopy(config.get("highlight_removal", {}))
    p.update(config.get("highlight_detection", {}))
    p["_runtime"] = config.get("runtime", {})
    return p


def _rotate_image_and_landmarks(image_bgr: np.ndarray, landmarks: np.ndarray, angle_deg: float, center) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(center), -angle_deg, 1.0)
    aligned = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    pts = landmarks[:, :2]
    ones = np.ones((len(pts), 1), dtype=np.float32)
    new_xy = np.hstack([pts, ones]) @ M.T
    aligned_lm = landmarks.copy()
    aligned_lm[:, :2] = new_xy
    invM = cv2.invertAffineTransform(M)
    return aligned, aligned_lm, M, invM


def _warp_mask(mask: np.ndarray, M: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    warped = cv2.warpAffine(mask_to_uint8(mask), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def _warp_image(image: np.ndarray, M: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _pipeline_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("pipeline", {}) or {}


def _detect_at_process_scale(image_bgr: np.ndarray, fd_params: Dict[str, Any], process_scale: float):
    process_scale = clamp_process_scale(process_scale)
    if process_scale >= 0.999:
        return detect_face(image_bgr, fd_params), 1.0
    small = resize_for_process(image_bgr, process_scale)
    det = detect_face(small, fd_params)
    if det.selected is None:
        return det, process_scale
    inv = 1.0 / process_scale
    selected = scale_face_to_shape(det.selected, inv, image_bgr.shape)
    all_faces = [scale_face_to_shape(f, inv, image_bgr.shape) for f in det.all_faces]
    return DetectOutput(selected, all_faces, det.status, det.warnings), process_scale


def _detect_highlight_masks(
    image_bgr: np.ndarray,
    face: FaceResult,
    config: Dict[str, Any],
) -> tuple[HighlightOutput, np.ndarray, np.ndarray, str, Dict[str, float]]:
    pipeline = _pipeline_cfg(config)
    face_scale, highlight_scale, is_compromise = resolve_process_scales(pipeline)
    refine = bool(pipeline.get("refine_upsampled_masks", True))
    hd_params = deepcopy(config.get("highlight_detection", {}))
    hd_params["_runtime"] = config.get("runtime", {})
    region_params = config.get("regions", {})
    stage_times: Dict[str, float] = {}

    # 1.0、折中+精修开、或非折中+精修开：高光在原图分辨率检测。
    use_full_highlight = highlight_scale >= 0.999 or refine
    if use_full_highlight:
        if is_compromise:
            t = time.perf_counter()
            small_face = resize_for_process(image_bgr, face_scale)
            lm_face = scale_landmarks(face.landmarks, face_scale)
            build_face_regions(small_face, lm_face, region_params)
            stage_times["区域生成(1/4)"] = time.perf_counter() - t
        t = time.perf_counter()
        regions = build_face_regions(image_bgr, face.landmarks, region_params)
        region_label = "区域生成" if highlight_scale >= 0.999 else "区域生成(全分辨率)"
        stage_times[region_label] = time.perf_counter() - t
        t = time.perf_counter()
        highlight = detect_highlight(image_bgr, regions, hd_params)
        if highlight_scale >= 0.999:
            stage_times["高光检测"] = time.perf_counter() - t
            stage_name = "高光检测"
        elif is_compromise:
            stage_times["高光检测(全分辨率)"] = time.perf_counter() - t
            stage_name = "折中(人脸1/4 + 高光原图)"
        else:
            stage_times["高光检测(全分辨率)"] = time.perf_counter() - t
            stage_name = f"高光检测(原分辨率, 人脸/区域比例 {face_scale:g})"
        return highlight, highlight.hard_mask, highlight.soft_mask, stage_name, stage_times, regions

    hl_scale = highlight_scale
    small = resize_for_process(image_bgr, hl_scale)
    lm_hl = scale_landmarks(face.landmarks, hl_scale)

    if is_compromise:
        t = time.perf_counter()
        small_face = resize_for_process(image_bgr, face_scale)
        lm_face = scale_landmarks(face.landmarks, face_scale)
        build_face_regions(small_face, lm_face, region_params)
        stage_times["区域生成(1/4)"] = time.perf_counter() - t

    t = time.perf_counter()
    regions_hl = build_face_regions(small, lm_hl, region_params)
    region_label = "区域生成(1/2)" if is_compromise else "区域生成(低分辨率)"
    stage_times[region_label] = time.perf_counter() - t

    t = time.perf_counter()
    hd_scaled = scaled_highlight_params(hd_params, hl_scale)
    hd_scaled["_runtime"] = config.get("runtime", {})
    highlight = detect_highlight(small, regions_hl, hd_scaled)
    hl_label = "高光检测(1/2)" if is_compromise else "高光检测(低分辨率)"
    stage_times[hl_label] = time.perf_counter() - t

    t = time.perf_counter()
    hard_full = upsample_mask_hard(highlight.hard_mask, image_bgr.shape)
    soft_full = upsample_mask_soft(highlight.soft_mask, image_bgr.shape)
    stage_times["mask上采样"] = time.perf_counter() - t

    t = time.perf_counter()
    regions = build_face_regions(image_bgr, face.landmarks, region_params)
    stage_times["区域生成(全分辨率)"] = time.perf_counter() - t

    highlight_full = HighlightOutput(hard_full, soft_full, highlight.debug, list(highlight.warnings))
    if is_compromise:
        stage_name = "折中(人脸1/4 + 高光1/2 → 原图)"
    else:
        stage_name = f"高光检测(压缩比例 {hl_scale:g})"
    return highlight_full, hard_full, soft_full, stage_name, stage_times, regions


def _removal_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = _merge_removal_params(config)
    pipeline = _pipeline_cfg(config)
    params["enable_roi_removal"] = bool(pipeline.get("enable_roi_removal", True))
    params["roi_padding_ratio"] = float(pipeline.get("roi_padding_ratio", 0.12))
    return params


def _enable_visualization(config: Dict[str, Any]) -> bool:
    return bool(_pipeline_cfg(config).get("enable_visualization", True))


def _blank_view(image_bgr: np.ndarray) -> np.ndarray:
    return np.zeros_like(image_bgr)


def _build_visual_outputs(
    image_bgr: np.ndarray,
    face: FaceResult,
    all_faces: List[FaceResult],
    regions: RegionMasks,
    config: Dict[str, Any],
) -> dict[str, np.ndarray]:
    show_index = bool(config.get("face_detection", {}).get("show_landmark_index", False))
    treatable = regions.masks.get("treatable_skin", regions.masks.get("forehead"))
    return {
        "landmark_view_bgr": draw_landmark_view(image_bgr, face, all_faces, show_index=show_index),
        "face_region_view_bgr": draw_region_view(image_bgr, regions, face),
        "skin_mask_bgr": mask_preview(regions.masks["skin"]),
        "protect_mask_bgr": mask_preview(regions.masks["protect"]),
        "nose_bridge_bgr": mask_preview(regions.masks["nose_bridge"]),
        "forehead_bgr": mask_preview(treatable if treatable is not None else regions.masks["forehead"]),
        "left_cheek_bgr": mask_preview(regions.masks["left_cheek"]),
        "right_cheek_bgr": mask_preview(regions.masks["right_cheek"]),
    }


def _compute_result_elapsed(stage_times: Dict[str, float]) -> float:
    """至去高光结果图：人脸检测 + 区域/高光 + 去高光修复（不含质量审计与可视化）。"""
    skip = {"可视化输出", "质量审计", "总流程", "_result_core", "除可视化用时"}
    total = 0.0
    for key, value in stage_times.items():
        if key in skip:
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _finalize_stage_times(stage_times: Dict[str, float]) -> None:
    stage_times.pop("_result_core", None)
    stage_times["除可视化用时"] = _compute_result_elapsed(stage_times)


def _process_core(image_bgr: np.ndarray, face: FaceResult, all_faces: List[FaceResult], config: Dict[str, Any]) -> PipelineOutput:
    stage_times: Dict[str, float] = {}

    highlight, hard_mask, soft_mask, highlight_stage, hl_times, regions = _detect_highlight_masks(image_bgr, face, config)
    stage_times.update(hl_times)

    t = time.perf_counter()
    removal = remove_highlight(image_bgr, hard_mask, soft_mask, regions, _removal_params(config))
    stage_times["去高光修复"] = time.perf_counter() - t

    t = time.perf_counter()
    metrics, metric_warnings = compute_metrics(image_bgr, removal.result_bgr, removal.applied_mask, regions, face, config.get("highlight_removal", {}))
    stage_times["质量审计"] = time.perf_counter() - t

    warnings = []
    warnings.extend(regions.warnings)
    warnings.extend(highlight.warnings)
    warnings.extend(removal.warnings)
    warnings.extend(metric_warnings)
    warnings = list(dict.fromkeys([w for w in warnings if w]))

    if _enable_visualization(config):
        t = time.perf_counter()
        views = _build_visual_outputs(image_bgr, face, all_faces, regions, config)
        stage_times["可视化输出"] = time.perf_counter() - t
    else:
        blank = _blank_view(image_bgr)
        views = {
            "landmark_view_bgr": blank,
            "face_region_view_bgr": blank,
            "skin_mask_bgr": blank,
            "protect_mask_bgr": blank,
            "nose_bridge_bgr": blank,
            "forehead_bgr": blank,
            "left_cheek_bgr": blank,
            "right_cheek_bgr": blank,
        }

    status = "处理成功" if not warnings else "处理完成，存在需要人工复核的警告"
    return PipelineOutput(
        success=True,
        status=status,
        warnings=warnings,
        result_bgr=removal.result_bgr,
        highlight_mask=hard_mask,
        soft_mask=soft_mask,
        diff_bgr=removal.diff_bgr,
        landmark_view_bgr=views["landmark_view_bgr"],
        face_region_view_bgr=views["face_region_view_bgr"],
        skin_mask_bgr=views["skin_mask_bgr"],
        protect_mask_bgr=views["protect_mask_bgr"],
        nose_bridge_bgr=views["nose_bridge_bgr"],
        forehead_bgr=views["forehead_bgr"],
        left_cheek_bgr=views["left_cheek_bgr"],
        right_cheek_bgr=views["right_cheek_bgr"],
        metrics=metrics,
        metrics_text=metrics_to_text(metrics, warnings),
        params=config,
        regions=regions,
        face=face,
        detection_count=len(all_faces),
        stage_times=stage_times,
    )


def process_image(image_bgr: np.ndarray, config: Dict[str, Any]) -> PipelineOutput:
    config = deepcopy(config)
    total_t0 = time.perf_counter()
    t = time.perf_counter()
    fd_params = deepcopy(config.get("face_detection", {}))
    fd_params["_runtime"] = config.get("runtime", {})
    face_scale, _, _ = resolve_process_scales(_pipeline_cfg(config))
    det, _ = _detect_at_process_scale(image_bgr, fd_params, face_scale)
    detect_time = time.perf_counter() - t
    if det.selected is None:
        blank = np.zeros_like(image_bgr)
        return PipelineOutput(
            success=False,
            status=det.status,
            warnings=det.warnings,
            result_bgr=image_bgr.copy(),
            highlight_mask=np.zeros(image_bgr.shape[:2], np.uint8),
            soft_mask=np.zeros(image_bgr.shape[:2], np.uint8),
            diff_bgr=blank,
            landmark_view_bgr=image_bgr.copy(),
            face_region_view_bgr=image_bgr.copy(),
            skin_mask_bgr=blank,
            protect_mask_bgr=blank,
            nose_bridge_bgr=blank,
            forehead_bgr=blank,
            left_cheek_bgr=blank,
            right_cheek_bgr=blank,
            metrics={},
            metrics_text=det.status,
            params=config,
            detection_count=len(det.all_faces),
            stage_times={"人脸检测": detect_time, "总流程": time.perf_counter() - total_t0},
        )

    warnings = list(det.warnings)
    face = det.selected
    if config.get("face_detection", {}).get("enable_alignment", False):
        try:
            geo = get_geometry(face.landmarks, image_bgr.shape)
            aligned_img, aligned_lm, _, invM = _rotate_image_and_landmarks(image_bgr, face.landmarks, geo.angle_deg, np.asarray(geo.eye_mid, dtype=np.float32))
            aligned_face = FaceResult(
                bbox=bbox_from_landmarks(aligned_lm, image_bgr.shape),
                confidence=face.confidence,
                landmarks=aligned_lm,
                detector=face.detector + "+轻量对齐",
                face_index=face.face_index,
            )
            aligned_out = _process_core(aligned_img, aligned_face, [aligned_face], config)
            # 将处理结果与所有 mask 反变换回原图。
            t_warp = time.perf_counter()
            mask_back = _warp_mask(aligned_out.soft_mask, invM, image_bgr.shape)
            hard_back = _warp_mask(aligned_out.highlight_mask, invM, image_bgr.shape)
            result_back = _warp_image(aligned_out.result_bgr, invM, image_bgr.shape)
            alpha = (mask_back.astype(np.float32) / 255.0)[..., None]
            result = (image_bgr.astype(np.float32) * (1 - alpha) + result_back.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

            warped_masks = {}
            for k, v in aligned_out.regions.masks.items():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    warped_masks[k] = _warp_mask(v, invM, image_bgr.shape)
                else:
                    warped_masks[k] = v
            regions_back = RegionMasks(warped_masks, get_geometry(face.landmarks, image_bgr.shape), aligned_out.regions.warnings)
            warp_time = time.perf_counter() - t_warp
            diff = cv2.absdiff(image_bgr, result)
            diff_vis = np.clip(diff.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
            t = time.perf_counter()
            metrics, metric_warnings = compute_metrics(image_bgr, result, hard_back, regions_back, face, config.get("highlight_removal", {}))
            align_extra_time = time.perf_counter() - t
            all_warnings = warnings + aligned_out.warnings + metric_warnings
            all_warnings = list(dict.fromkeys([w for w in all_warnings if w]))
            stage_times = dict(aligned_out.stage_times)
            stage_times["反变换"] = warp_time
            stage_times["质量审计"] = float(stage_times.get("质量审计", 0.0)) + align_extra_time
            stage_times["人脸检测"] = detect_time
            if _enable_visualization(config):
                t_viz = time.perf_counter()
                views = _build_visual_outputs(image_bgr, face, det.all_faces, regions_back, config)
                stage_times["可视化输出"] = time.perf_counter() - t_viz
            else:
                blank = _blank_view(image_bgr)
                views = {k: blank for k in (
                    "landmark_view_bgr", "face_region_view_bgr", "skin_mask_bgr", "protect_mask_bgr",
                    "nose_bridge_bgr", "forehead_bgr", "left_cheek_bgr", "right_cheek_bgr",
                )}
            _finalize_stage_times(stage_times)
            stage_times["总流程"] = time.perf_counter() - total_t0
            return PipelineOutput(
                success=True,
                status="处理成功（已启用轻量对齐并反变换回原图）" if not all_warnings else "处理完成，存在需要人工复核的警告",
                warnings=all_warnings,
                result_bgr=result,
                highlight_mask=hard_back,
                soft_mask=mask_back,
                diff_bgr=diff_vis,
                landmark_view_bgr=views["landmark_view_bgr"],
                face_region_view_bgr=views["face_region_view_bgr"],
                skin_mask_bgr=views["skin_mask_bgr"],
                protect_mask_bgr=views["protect_mask_bgr"],
                nose_bridge_bgr=views["nose_bridge_bgr"],
                forehead_bgr=views["forehead_bgr"],
                left_cheek_bgr=views["left_cheek_bgr"],
                right_cheek_bgr=views["right_cheek_bgr"],
                metrics=metrics,
                metrics_text=metrics_to_text(metrics, all_warnings),
                params=config,
                regions=regions_back,
                face=face,
                detection_count=len(det.all_faces),
                stage_times=stage_times,
            )
        except Exception as exc:  # 安全回退，不让前端崩溃。
            warnings.append(f"轻量对齐失败，已回退到原图坐标处理：{exc}")

    out = _process_core(image_bgr, face, det.all_faces, config)
    stage_times = {"人脸检测": detect_time, **out.stage_times}
    _finalize_stage_times(stage_times)
    stage_times["总流程"] = time.perf_counter() - total_t0
    out.stage_times = stage_times
    out.warnings = list(dict.fromkeys(warnings + out.warnings))
    out.metrics_text = metrics_to_text(out.metrics, out.warnings)
    if warnings:
        out.status = "处理完成，存在需要人工复核的警告"
    return out


def update_config_from_flat(base: Dict[str, Any], flat: Dict[str, Any]) -> Dict[str, Any]:
    patch: Dict[str, Any] = {
        "face_detection": {},
        "regions": {},
        "highlight_detection": {},
        "highlight_removal": {},
        "pipeline": {},
    }
    for k, v in flat.items():
        if k.startswith("fd__"):
            patch["face_detection"][k[4:]] = v
        elif k.startswith("rg__"):
            patch["regions"][k[4:]] = v
        elif k.startswith("hd__"):
            patch["highlight_detection"][k[4:]] = v
        elif k.startswith("rm__"):
            patch["highlight_removal"][k[4:]] = v
        elif k.startswith("pl__"):
            patch["pipeline"][k[4:]] = v
    return recursive_update(base, patch)
