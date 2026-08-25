"""误差来源隔离对拍：把「Python vs 单 ONNX(C++ 内核)」的残差拆成可量化的独立来源。

对每张 data/*.png 报告四组数字（全部实测，不做推断）：
  1. 关键点：MediaPipe Tasks vs ORT landmarker（tools/onnx_landmarker_ref.py，
     与 C++ face_landmarker_ort.cpp 同一算法/同一 ONNX 权重），在流水线实际
     检测分辨率（1/4 图）上的 478 点 xy 最大/平均像素误差（已换算回原图像素）。
  2. 全链路：Python process_image（MediaPipe）vs 单 ONNX 会话（C++ 内核）。
  3. 关键点隔离：Python process_image（MediaPipe） vs Python process_image
     （关键点替换为 ORT landmarker，CV 代码完全相同）→ 只剩关键点来源的差异。
  4. CV 隔离：Python process_image（ORT 关键点） vs 单 ONNX 会话（C++ 内核）
     → 关键点近似相同，剩余差异来自 OpenCV 版本 / C++ 移植的浮点路径。

另外量化两件事：
  - Python process_image 自身的确定性（同图同配置跑两遍是否逐位相同）；
  - CLI「常用模式」与 configs/default.yaml 原始值的配置差异
    （目前唯一差异 highlight_detection.brow_region_max_fraction 0.32 vs 0.28）
    是否影响任何一张样例图的输出。

结果打印表格，并把标记章节写入 tools/PARITY_RESULTS.md。

用法：
  python tools/parity_isolate.py                 # 全部 17 张
  python tools/parity_isolate.py --images data/1.png
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy  # noqa: E402

bootstrap_before_numpy()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from cli_process import load_cli_config, preload_models  # noqa: E402
from highlight_removal import face_detect  # noqa: E402
from highlight_removal.face_detect import FaceResult, detect_face, select_face  # noqa: E402
from highlight_removal.face_landmarks import bbox_from_landmarks  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import load_yaml, resize_for_process  # noqa: E402
from onnx_landmarker_ref import OnnxFaceLandmarker  # noqa: E402

SECTION_BEGIN = "<!-- parity-isolate-begin -->"
SECTION_END = "<!-- parity-isolate-end -->"


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"mae": float(d.mean()), "max": int(d.max()), "pct": float((d.max(axis=2) > 0).mean() * 100.0)}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a1, b1 = a > 0, b > 0
    union = int(np.logical_or(a1, b1).sum())
    return 1.0 if union == 0 else float(np.logical_and(a1, b1).sum() / union)


def default_yaml_config() -> dict:
    """与 models/high_removal.onnx 内嵌配置（configs/default.yaml 原始值）等价的运行配置。"""
    from highlight_removal.utils import apply_runtime_mode

    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    cfg["runtime"] = apply_runtime_mode("CPU狂暴模式")
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


class OrtTasksPatch:
    """把 face_detect._detect_mediapipe_tasks 换成 ORT landmarker（与 C++ 同算法）。"""

    def __init__(self, landmarker: OnnxFaceLandmarker):
        self.landmarker = landmarker
        self._orig = None

    def _detect(self, image_bgr: np.ndarray, params: dict):
        min_conf = float(params.get("face_detection_confidence", 0.55))
        results = []
        for idx, (px, _score) in enumerate(self.landmarker.detect_pixels(image_bgr)):
            bbox = bbox_from_landmarks(px, image_bgr.shape)
            results.append(
                FaceResult(bbox=bbox, confidence=max(min_conf, 0.90), landmarks=px,
                           detector="OrtFaceLandmarker", face_index=idx)
            )
        return results

    def __enter__(self):
        self._orig = face_detect._detect_mediapipe_tasks
        face_detect._detect_mediapipe_tasks = self._detect
        return self

    def __exit__(self, *exc):
        face_detect._detect_mediapipe_tasks = self._orig
        return False


def landmark_error(img: np.ndarray, cfg: dict, ort_lm: OnnxFaceLandmarker) -> tuple[float, float]:
    """1/4 检测分辨率上选中人脸的 478 点 xy 误差（换算回原图像素）：(max, mean)。"""
    fd = dict(cfg.get("face_detection", {}))
    fd["_runtime"] = cfg.get("runtime", {})
    small = resize_for_process(img, 0.25)
    det = detect_face(small, fd)
    if det.selected is None or det.selected.landmarks is None:
        return float("nan"), float("nan")
    mp_lms = det.selected.landmarks[:, :2].astype(np.float64)

    min_conf = float(fd.get("face_detection_confidence", 0.55))
    candidates = []
    for idx, (px, _score) in enumerate(ort_lm.detect_pixels(small)):
        bbox = bbox_from_landmarks(px, small.shape)
        candidates.append(FaceResult(bbox, max(min_conf, 0.90), px, "OrtFaceLandmarker", idx))
    sel = select_face(candidates, fd.get("multi_face_strategy", "最大人脸"), int(fd.get("manual_face_index", 0)))
    if sel is None or sel.landmarks is None:
        return float("nan"), float("nan")
    ort_lms = sel.landmarks[:, :2].astype(np.float64)

    err = np.abs(mp_lms - ort_lms) * 4.0  # 1/4 图坐标 → 原图像素
    per_point = np.linalg.norm(mp_lms - ort_lms, axis=1) * 4.0
    return float(per_point.max()), float(per_point.mean())


def fmt(m: dict) -> str:
    return f"{m['mae']:.4f}/{m['max']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "high_removal.onnx"))
    parser.add_argument("--ops", default=str(ROOT / "cpp" / "build" / "libhigh_removal_ops.so"))
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--report", default=str(ROOT / "tools" / "PARITY_RESULTS.md"))
    args = parser.parse_args()

    images = [Path(p) for p in (args.images or sorted((ROOT / "data").glob("*.png"), key=lambda p: (len(p.stem), p.stem)))]

    cfg = default_yaml_config()
    if not preload_models(cfg):
        return 3
    fd = cfg.get("face_detection", {})
    ort_lm = OnnxFaceLandmarker(
        ROOT / "models",
        num_faces=int(fd.get("max_faces", 4)),
        min_detection_confidence=float(fd.get("face_detection_confidence", 0.55)),
        min_presence_confidence=float(fd.get("landmark_detection_confidence", 0.50)),
    )

    so = ort.SessionOptions()
    so.register_custom_ops_library(args.ops)
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])

    # 确定性检查：同图同配置跑两遍 Python process_image。
    probe = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    o1 = process_image(probe, cfg)
    o2 = process_image(probe, cfg)
    deterministic = bool(
        np.array_equal(o1.result_bgr, o2.result_bgr) and np.array_equal(o1.highlight_mask, o2.highlight_mask)
    )

    # 配置差异（CLI 常用模式 vs default.yaml 原始值）是否影响输出。
    cli_cfg = load_cli_config("常用模式")
    cfg_diff_images = []

    rows = []
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        lm_max, lm_mean = landmark_error(img, cfg, ort_lm)

        py_mp = process_image(img, cfg)
        onnx_result, onnx_hard = sess.run(["result", "hard_mask"], {"image": np.ascontiguousarray(img)})
        with OrtTasksPatch(ort_lm):
            py_ort = process_image(img, cfg)
        py_cli = process_image(img, cli_cfg)
        if not np.array_equal(py_cli.result_bgr, py_mp.result_bgr):
            cfg_diff_images.append(path.name)

        m_full = metrics(py_mp.result_bgr, onnx_result)
        m_full["iou"] = mask_iou(py_mp.highlight_mask, onnx_hard)
        m_lm = metrics(py_mp.result_bgr, py_ort.result_bgr)
        m_lm["iou"] = mask_iou(py_mp.highlight_mask, py_ort.highlight_mask)
        m_cv = metrics(py_ort.result_bgr, onnx_result)
        m_cv["iou"] = mask_iou(py_ort.highlight_mask, onnx_hard)
        rows.append((path.name, lm_max, lm_mean, m_full, m_lm, m_cv))
        print(
            f"{path.name}: lm_max={lm_max:.3f}px lm_mean={lm_mean:.4f}px | "
            f"全链路 {fmt(m_full)} IoU={m_full['iou']:.4f} | "
            f"仅关键点 {fmt(m_lm)} IoU={m_lm['iou']:.4f} | "
            f"仅CV {fmt(m_cv)} IoU={m_cv['iou']:.4f}"
        )

    def avg(key_fn):
        return float(np.mean([key_fn(r) for r in rows]))

    summary = (
        f"平均关键点误差 max={avg(lambda r: r[1]):.3f}px / mean={avg(lambda r: r[2]):.4f}px；"
        f"全链路 MAE={avg(lambda r: r[3]['mae']):.4f}，仅关键点 MAE={avg(lambda r: r[4]['mae']):.4f}，"
        f"仅CV MAE={avg(lambda r: r[5]['mae']):.4f}；"
        f"Python 自身确定性（同图两遍逐位相同）：{'是' if deterministic else '否'}；"
        f"CLI常用模式 vs default.yaml 配置差异影响的图片：{len(cfg_diff_images)}/{len(rows)}"
        + (f"（{'、'.join(cfg_diff_images)}）" if cfg_diff_images else "")
    )
    print("\n" + summary)

    date = datetime.date.today().isoformat()
    lines = [
        SECTION_BEGIN,
        f"## 误差来源隔离（{date}，`python tools/parity_isolate.py`）",
        "",
        "把「Python(MediaPipe) vs 单 ONNX(C++ 内核)」的残差拆解为独立来源。",
        "「仅关键点」= 两边都跑 pip OpenCV 的 Python CV 代码，只把关键点从 MediaPipe 换成 ORT landmarker",
        "（与 C++ 内核同算法、同 ONNX 权重）；「仅 CV」= 关键点近似相同后，剩余差异来自 OpenCV 版本",
        "（pip 4.14 vs apt 4.6）与 C++ 浮点路径。关键点误差为 1/4 检测分辨率上的 478 点欧氏距离（换算回原图像素）。",
        "",
        "| 图片 | 关键点 max(px) | 关键点 mean(px) | 全链路 MAE/max | 全链路 IoU | 仅关键点 MAE/max | 仅关键点 IoU | 仅CV MAE/max | 仅CV IoU |",
        "|------|--------------|----------------|----------------|-----------|-----------------|-------------|--------------|----------|",
    ]
    for name, lm_max, lm_mean, m_full, m_lm, m_cv in rows:
        lines.append(
            f"| {name} | {lm_max:.3f} | {lm_mean:.4f} | {fmt(m_full)} | {m_full['iou']:.4f} "
            f"| {fmt(m_lm)} | {m_lm['iou']:.4f} | {fmt(m_cv)} | {m_cv['iou']:.4f} |"
        )
    lines += ["", f"**汇总**：{summary}", SECTION_END]
    section = "\n".join(lines)

    report = Path(args.report)
    text = report.read_text(encoding="utf-8") if report.is_file() else "# 输出对拍结果\n"
    if SECTION_BEGIN in text and SECTION_END in text:
        head, rest = text.split(SECTION_BEGIN, 1)
        _, tail = rest.split(SECTION_END, 1)
        text = head.rstrip() + "\n\n" + section + tail
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    report.write_text(text, encoding="utf-8")
    print(f"\n报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
