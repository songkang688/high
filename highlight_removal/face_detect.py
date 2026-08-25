

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_LANDMARKER_CACHE: Dict[tuple, Any] = {}
_LANDMARKER_CACHE_LOCK = Lock()
_LAST_LANDMARKER_ERROR = ""

from .face_landmarks import (
    FACE_OVAL,
    INNER_LIPS,
    LEFT_BROW,
    LEFT_EYE,
    MOUTH_CORNERS,
    NOSE_BRIDGE,
    NOSE_TIP,
    NOSE_WING,
    OUTER_LIPS,
    RIGHT_BROW,
    RIGHT_EYE,
    bbox_from_landmarks,
)
from .utils import bbox_area, iou_bbox, put_text_cn


@dataclass
class FaceResult:
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray]
    detector: str
    face_index: int = 0
    warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["landmarks"] = None if self.landmarks is None else f"{len(self.landmarks)} points"
        return d


@dataclass
class DetectOutput:
    selected: Optional[FaceResult]
    all_faces: List[FaceResult]
    status: str
    warnings: List[str]


def _resolve_path(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # 项目根目录：highlight_removal/..。
    root = Path(__file__).resolve().parents[1]
    p2 = root / p
    if p2.exists():
        return p2
    return None


def _mediapipe_face_detection_scores(image_rgb: np.ndarray, min_conf: float, max_faces: int) -> List[FaceResult]:
    try:
        import mediapipe as mp  # pylint: disable=import-outside-toplevel
    except Exception:
        return []
    if not hasattr(mp, "solutions"):
        return []
    h, w = image_rgb.shape[:2]
    out: List[FaceResult] = []
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=float(min_conf)) as det:
        res = det.process(image_rgb)
        if not res.detections:
            return []
        for idx, d in enumerate(res.detections[:max_faces]):
            rb = d.location_data.relative_bounding_box
            x = int(max(0, round(rb.xmin * w)))
            y = int(max(0, round(rb.ymin * h)))
            bw = int(min(w - x, round(rb.width * w)))
            bh = int(min(h - y, round(rb.height * h)))
            score = float(d.score[0]) if d.score else 0.0
            out.append(FaceResult((x, y, bw, bh), score, None, "MediaPipeFaceDetection", idx))
    return out


def _detect_mediapipe_legacy(image_bgr: np.ndarray, params: Dict[str, Any]) -> List[FaceResult]:
    try:
        import mediapipe as mp  # pylint: disable=import-outside-toplevel
    except Exception:
        return []
    if not hasattr(mp, "solutions"):
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    max_faces = int(params.get("max_faces", 4))
    min_conf = float(params.get("face_detection_confidence", 0.55))
    lm_conf = float(params.get("landmark_detection_confidence", 0.50))
    det_faces = _mediapipe_face_detection_scores(image_rgb, min_conf, max_faces)
    results: List[FaceResult] = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=min_conf,
        min_tracking_confidence=lm_conf,
    ) as face_mesh:
        res = face_mesh.process(image_rgb)
        if not res.multi_face_landmarks:
            return []
        h, w = image_bgr.shape[:2]
        for idx, face_lms in enumerate(res.multi_face_landmarks):
            pts = []
            for lm in face_lms.landmark:
                pts.append([lm.x * w, lm.y * h, lm.z * max(w, h)])
            lms = np.asarray(pts, dtype=np.float32)
            bbox = bbox_from_landmarks(lms, image_bgr.shape)
            conf = 0.99
            for df in det_faces:
                if iou_bbox(bbox, df.bbox) > 0.15:
                    conf = max(conf if det_faces else 0.0, df.confidence)
            results.append(FaceResult(bbox=bbox, confidence=float(conf), landmarks=lms, detector="MediaPipeFaceMesh", face_index=idx))
    return results


def _landmarker_cache_key(params: Dict[str, Any]) -> Optional[tuple]:
    model_path = _resolve_path(params.get("mediapipe_task_model_path", "models/face_landmarker.task"))
    if model_path is None:
        return None
    runtime = params.get("_runtime", {}) or {}
    return (
        str(model_path),
        bool(runtime.get("gpu_enabled")),
        int(params.get("max_faces", 4)),
        float(params.get("face_detection_confidence", 0.55)),
        float(params.get("landmark_detection_confidence", 0.50)),
    )


def _create_face_landmarker(params: Dict[str, Any]):
    global _LAST_LANDMARKER_ERROR
    model_path = _resolve_path(params.get("mediapipe_task_model_path", "models/face_landmarker.task"))
    if model_path is None:
        _LAST_LANDMARKER_ERROR = (
            f"找不到模型文件: {params.get('mediapipe_task_model_path', 'models/face_landmarker.task')}"
        )
        return None, "CPU"
    try:
        from mediapipe.tasks import python  # pylint: disable=import-outside-toplevel
        from mediapipe.tasks.python import vision  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        _LAST_LANDMARKER_ERROR = f"mediapipe 导入失败: {exc}"
        return None, "CPU"
    max_faces = int(params.get("max_faces", 4))
    min_conf = float(params.get("face_detection_confidence", 0.55))
    lm_conf = float(params.get("landmark_detection_confidence", 0.50))
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=lm_conf,
        min_tracking_confidence=lm_conf,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        _LAST_LANDMARKER_ERROR = f"模型加载失败 ({model_path}): {exc}"
        return None, "CPU"
    _LAST_LANDMARKER_ERROR = ""
    return landmarker, "CPU"


def _get_face_landmarker(params: Dict[str, Any]):
    key = _landmarker_cache_key(params)
    if key is None:
        return None, "CPU"
    with _LANDMARKER_CACHE_LOCK:
        cached = _LANDMARKER_CACHE.get(key)
        if cached is not None:
            return cached
        landmarker, delegate_name = _create_face_landmarker(params)
        if landmarker is None:
            return None, delegate_name
        _LANDMARKER_CACHE[key] = (landmarker, delegate_name)
        return landmarker, delegate_name


def preload_face_landmarker(params: Dict[str, Any]) -> bool:
    """启动时预加载 FaceLandmarker，后续每张图复用同一实例。"""
    landmarker, _ = _get_face_landmarker(params)
    return landmarker is not None


def get_last_landmarker_error() -> str:
    return _LAST_LANDMARKER_ERROR


def scale_face_to_shape(face: FaceResult, inv_scale: float, shape) -> FaceResult:
    """将低分辨率坐标系下的人脸结果映射回原图坐标系。"""
    if face.landmarks is None or abs(inv_scale - 1.0) < 1e-6:
        return face
    lm = face.landmarks.copy()
    lm[:, :2] *= inv_scale
    if lm.shape[1] > 2:
        lm[:, 2] *= inv_scale
    x, y, bw, bh = face.bbox
    bbox = (
        int(round(x * inv_scale)),
        int(round(y * inv_scale)),
        int(round(bw * inv_scale)),
        int(round(bh * inv_scale)),
    )
    bbox = bbox_from_landmarks(lm, shape) if lm is not None else bbox
    return FaceResult(
        bbox=bbox,
        confidence=face.confidence,
        landmarks=lm,
        detector=face.detector,
        face_index=face.face_index,
        warning=face.warning,
    )


def _detect_mediapipe_tasks(image_bgr: np.ndarray, params: Dict[str, Any]) -> List[FaceResult]:
    try:
        import mediapipe as mp  # pylint: disable=import-outside-toplevel
    except Exception:
        return []
    landmarker, delegate_name = _get_face_landmarker(params)
    if landmarker is None:
        return []
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    min_conf = float(params.get("face_detection_confidence", 0.55))
    results: List[FaceResult] = []
    h, w = image_bgr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return []
    for idx, lms_norm in enumerate(res.face_landmarks):
        pts = []
        for lm in lms_norm:
            pts.append([lm.x * w, lm.y * h, lm.z * max(w, h)])
        lms = np.asarray(pts, dtype=np.float32)
        bbox = bbox_from_landmarks(lms, image_bgr.shape)
        results.append(
            FaceResult(
                bbox=bbox,
                confidence=max(min_conf, 0.90),
                landmarks=lms,
                detector=f"MediaPipeFaceLandmarker-{delegate_name}",
                face_index=idx,
            )
        )
    return results


def detect_mediapipe(image_bgr: np.ndarray, params: Dict[str, Any]) -> List[FaceResult]:
    # 第一优先级：MediaPipe Tasks FaceLandmarker（如模型存在）。
    faces = _detect_mediapipe_tasks(image_bgr, params)
    if faces:
        return faces
    # 第二尝试：老版 MediaPipe solutions FaceMesh（适配旧环境）。
    return _detect_mediapipe_legacy(image_bgr, params)


def detect_haar(image_bgr: np.ndarray, params: Dict[str, Any]) -> List[FaceResult]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    scale_factor = float(params.get("haar_scale_factor", 1.08))
    min_neighbors = int(params.get("haar_min_neighbors", 5))
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(40, 40))
    out: List[FaceResult] = []
    for idx, (x, y, w, h) in enumerate(faces):
        out.append(FaceResult((int(x), int(y), int(w), int(h)), 0.60, None, "OpenCVHaar", idx, "fallback 检测到人脸框，但未获得可靠关键点"))
    return out


def _assign_cycle(arr: np.ndarray, mp_indices: List[int], src_points: np.ndarray) -> None:
    if len(src_points) == 0:
        return
    for k, idx in enumerate(mp_indices):
        arr[idx, :2] = src_points[k % len(src_points)]
        arr[idx, 2] = 0.0


def _convert_lbf68_to_mediapipe_like(lm68: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """将 OpenCV LBF 的 68 点转换为本工程使用的 MediaPipe 语义索引子集。

    转换只服务于 fallback：五官点来自真实 68 点；额头/上轮廓由检测 bbox、眉毛和下颌点推断。
    它不是固定图像中心，也不使用固定图像坐标。
    """
    arr = np.full((478, 3), np.nan, dtype=np.float32)
    lm68 = lm68.astype(np.float32).reshape(-1, 2)
    x, y, bw, bh = bbox
    brow = lm68[17:27]
    jaw = lm68[0:17]
    brow_y = float(np.min(brow[:, 1]))
    brow_left = brow[0]
    brow_right = brow[-1]
    top_y = max(0.0, min(float(y + 0.05 * bh), brow_y - 0.28 * bh))
    top = np.array([(brow_left[0] + brow_right[0]) / 2.0, top_y], dtype=np.float32)
    left_tem = np.array([float(x + 0.12 * bw), float(y + 0.22 * bh)], dtype=np.float32)
    right_tem = np.array([float(x + 0.88 * bw), float(y + 0.22 * bh)], dtype=np.float32)
    left_fore = np.array([float(x + 0.28 * bw), float(y + 0.10 * bh)], dtype=np.float32)
    right_fore = np.array([float(x + 0.72 * bw), float(y + 0.10 * bh)], dtype=np.float32)

    # 按 MediaPipe FACE_OVAL 顺序构造闭合轮廓：顶部 -> 图像右侧 -> 下巴 -> 图像左侧 -> 顶部。
    oval_coords = [
        top, right_fore, brow_right, right_tem, jaw[16], jaw[15], jaw[14], jaw[13], jaw[12], jaw[11], jaw[10], jaw[9],
        jaw[8], jaw[8], jaw[8], jaw[8], jaw[8], jaw[8], jaw[8], jaw[7], jaw[6], jaw[5], jaw[4], jaw[3],
        jaw[2], jaw[1], jaw[0], jaw[0], jaw[0], left_tem, brow_left, left_fore, top, top, top, top,
    ]
    for idx, p in zip(FACE_OVAL, oval_coords):
        arr[idx, :2] = p
        arr[idx, 2] = 0.0

    _assign_cycle(arr, LEFT_EYE, lm68[36:42])
    _assign_cycle(arr, RIGHT_EYE, lm68[42:48])
    _assign_cycle(arr, LEFT_BROW, lm68[17:22])
    _assign_cycle(arr, RIGHT_BROW, lm68[22:27])
    _assign_cycle(arr, OUTER_LIPS, lm68[48:60])
    _assign_cycle(arr, INNER_LIPS, lm68[60:68])
    _assign_cycle(arr, NOSE_BRIDGE, lm68[27:31])
    _assign_cycle(arr, NOSE_TIP, lm68[[30, 31, 32, 33, 34, 35]])
    _assign_cycle(arr, NOSE_WING, lm68[31:36])
    arr[4, :2] = lm68[30]
    arr[5, :2] = lm68[30]
    arr[1, :2] = lm68[30]
    arr[2, :2] = lm68[33]
    arr[61, :2] = lm68[48]
    arr[291, :2] = lm68[54]
    arr[152, :2] = lm68[8]
    arr[152, 2] = 0.0
    # 下巴和脸颊相关点补充
    for idx, p in zip([148, 176, 149, 150, 136, 172, 58], [lm68[7], lm68[6], lm68[5], lm68[4], lm68[3], lm68[2], lm68[1]]):
        arr[idx, :2] = p
        arr[idx, 2] = 0
    for idx, p in zip([378, 400, 377, 365, 379], [lm68[9], lm68[10], lm68[11], lm68[12], lm68[13]]):
        arr[idx, :2] = p
        arr[idx, 2] = 0
    return arr


def detect_lbf_facemark(image_bgr: np.ndarray, params: Dict[str, Any]) -> List[FaceResult]:
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        return []
    model = _resolve_path(params.get("lbf_model_path", "models/lbfmodel.yaml"))
    if model is None:
        return []
    boxes = detect_haar(image_bgr, params)
    if not boxes:
        return []
    faces_np = np.array([b.bbox for b in boxes], dtype=np.int32)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(model))
    ok, landmarks = facemark.fit(image_bgr, faces_np)
    if not ok or landmarks is None:
        return []
    out: List[FaceResult] = []
    for idx, lm in enumerate(landmarks):
        lm68 = lm.reshape(-1, 2).astype(np.float32)
        bbox = tuple(int(v) for v in boxes[idx].bbox)
        mp_like = _convert_lbf68_to_mediapipe_like(lm68, bbox)
        bbox2 = bbox_from_landmarks(mp_like, image_bgr.shape)
        out.append(FaceResult(bbox2, max(0.66, boxes[idx].confidence), mp_like, "OpenCVLBF68Fallback", idx))
    return out


def _try_landmarks_on_roi(image_bgr: np.ndarray, face: FaceResult, params: Dict[str, Any]) -> FaceResult:
    x, y, w, h = face.bbox
    ih, iw = image_bgr.shape[:2]
    pad = int(max(w, h) * 0.35)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(iw, x + w + pad)
    y1 = min(ih, y + h + pad)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return face
    relaxed = dict(params)
    relaxed["face_detection_confidence"] = min(0.35, float(params.get("face_detection_confidence", 0.55)))
    relaxed["landmark_detection_confidence"] = min(0.35, float(params.get("landmark_detection_confidence", 0.50)))
    relaxed["max_faces"] = 1
    faces = detect_mediapipe(crop, relaxed) if params.get("enable_mediapipe", True) else []
    if not faces or faces[0].landmarks is None:
        face.warning += "；关键点 fallback 失败"
        return face
    lms = faces[0].landmarks.copy()
    lms[:, 0] += x0
    lms[:, 1] += y0
    bbox = bbox_from_landmarks(lms, image_bgr.shape)
    return FaceResult(bbox, max(face.confidence, 0.65), lms, face.detector + "+MediaPipeROI", face.face_index)


def select_face(faces: List[FaceResult], strategy: str = "最大人脸", manual_index: int = 0) -> Optional[FaceResult]:
    valid = [f for f in faces if f.bbox[2] > 0 and f.bbox[3] > 0]
    if not valid:
        return None
    if strategy == "手动编号":
        manual_index = int(np.clip(manual_index, 0, len(valid) - 1))
        return valid[manual_index]
    if strategy == "置信度最高":
        return max(valid, key=lambda f: f.confidence)
    return max(valid, key=lambda f: (bbox_area(f.bbox), f.confidence))


def detect_face(image_bgr: np.ndarray, params: Dict[str, Any]) -> DetectOutput:
    warnings: List[str] = []
    all_faces: List[FaceResult] = []

    if params.get("enable_mediapipe", True):
        try:
            mp_faces = detect_mediapipe(image_bgr, params)
            all_faces.extend([f for f in mp_faces if f.landmarks is not None])
            if not mp_faces:
                warnings.append("MediaPipe 未返回可靠人脸关键点，正在尝试 fallback 检测器")
        except Exception as exc:
            warnings.append(f"MediaPipe 检测失败，正在尝试 fallback 检测器：{exc}")

    if not all_faces and params.get("enable_fallback_detector", True):
        lbf_faces = detect_lbf_facemark(image_bgr, params)
        if lbf_faces:
            warnings.append("已使用 OpenCV LBF 68 点 fallback；建议在生产环境配置 MediaPipe FaceLandmarker 模型以获得更高精度")
            all_faces.extend(lbf_faces)
        else:
            fb = detect_haar(image_bgr, params)
            fb_with_lm = [_try_landmarks_on_roi(image_bgr, f, params) for f in fb]
            all_faces.extend(fb_with_lm)

    if not all_faces:
        return DetectOutput(None, [], "未检测到可靠人脸，请更换图片或调整检测参数", warnings)

    strategy = params.get("multi_face_strategy", "最大人脸")
    manual_index = int(params.get("manual_face_index", 0))
    selected = select_face(all_faces, strategy, manual_index)
    if selected is None:
        return DetectOutput(None, all_faces, "未检测到可靠人脸，请更换图片或调整检测参数", warnings)

    if len(all_faces) > 1:
        warnings.append("检测到多张人脸，已默认处理最大人脸" if strategy == "最大人脸" else f"检测到多张人脸，已按{strategy}处理")

    min_conf = float(params.get("face_detection_confidence", 0.55))
    if selected.confidence < min_conf:
        warnings.append("人脸检测置信度较低，请检查定位结果或调节检测阈值")

    if selected.landmarks is None or len(selected.landmarks) < 300:
        return DetectOutput(None, all_faces, "未检测到可靠人脸关键点，请更换图片或调低检测阈值", warnings)

    return DetectOutput(selected, all_faces, "人脸检测成功", warnings)


def draw_faces(image_bgr: np.ndarray, faces: List[FaceResult], selected: Optional[FaceResult] = None) -> np.ndarray:
    out = image_bgr.copy()
    for idx, f in enumerate(faces):
        x, y, w, h = f.bbox
        is_selected = selected is not None and f.face_index == selected.face_index and f.detector == selected.detector
        color = (0, 255, 0) if is_selected else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        out = put_text_cn(out, f"人脸{idx} 置信度{f.confidence:.2f}", (x, max(0, y - 8)), color, font_size=18)
    return out
