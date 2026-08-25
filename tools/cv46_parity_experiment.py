"""实验 B：验证「C++ 内核残余差异的剩余杠杆 = OpenCV 版本」。

在 pip OpenCV 4.6.0（与 C++ 侧 apt libopencv 4.6.0 同版本、numpy<2 的隔离 venv）里，
用 ORT landmarker（与 C++ 内核同算法、同 ONNX 权重）替换 MediaPipe 关键点后运行
原始 Python 流水线 process_image，与预先保存的单 ONNX C++ 内核输出
（runtime_outputs/detection_explore/cppkernel_*_result.png / _mask.png）逐像素对拍。

若 MAE 相比主环境（pip OpenCV 4.14）的「仅CV」残差（≈0.0026）显著下降，
即实测证明：关键点对齐后，剩余差异主要来自 OpenCV 版本差异，
「换用与 C++ 侧相同的 OpenCV 版本」是 C++ 内核逼近逐位一致的最后杠杆。

运行（需先在主环境保存 C++ 内核输出，见 docs/FINAL_DETECTION_REPORT.md）：
  /tmp/cv46venv/bin/python tools/cv46_parity_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy  # noqa: E402

bootstrap_before_numpy()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from highlight_removal import face_detect  # noqa: E402
from highlight_removal.face_detect import FaceResult  # noqa: E402
from highlight_removal.face_landmarks import bbox_from_landmarks  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE  # noqa: E402
from highlight_removal.utils import apply_runtime_mode, load_yaml  # noqa: E402
from onnx_landmarker_ref import OnnxFaceLandmarker  # noqa: E402

REF_DIR = ROOT / "runtime_outputs" / "detection_explore"


def main() -> int:
    print(f"cv2 {cv2.__version__}  numpy {np.__version__}")
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False

    fd = cfg.get("face_detection", {})
    ort_lm = OnnxFaceLandmarker(
        ROOT / "models",
        num_faces=int(fd.get("max_faces", 4)),
        min_detection_confidence=float(fd.get("face_detection_confidence", 0.55)),
        min_presence_confidence=float(fd.get("landmark_detection_confidence", 0.50)),
    )
    min_conf = float(fd.get("face_detection_confidence", 0.55))

    def _detect(image_bgr, params):
        results = []
        for idx, (px, _score) in enumerate(ort_lm.detect_pixels(image_bgr)):
            bbox = bbox_from_landmarks(px, image_bgr.shape)
            results.append(FaceResult(bbox, max(min_conf, 0.90), px, "OrtFaceLandmarker", idx))
        return results

    face_detect._detect_mediapipe_tasks = _detect  # 关键点与 C++ 内核完全同源

    for stem in ("1", "15", "16"):
        img = cv2.imread(str(ROOT / "data" / f"{stem}.png"), cv2.IMREAD_COLOR)
        ref_result = cv2.imread(str(REF_DIR / f"cppkernel_{stem}_result.png"), cv2.IMREAD_COLOR)
        ref_mask = cv2.imread(str(REF_DIR / f"cppkernel_{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or ref_result is None or ref_mask is None:
            print(f"[跳过] {stem}: 缺少输入或 C++ 内核参考输出")
            continue
        out = process_image(img, cfg)
        if not out.success:
            print(f"[失败] {stem}: {out.status}")
            continue
        d = np.abs(out.result_bgr.astype(np.int16) - ref_result.astype(np.int16))
        a1, b1 = out.highlight_mask > 0, ref_mask > 0
        union = int(np.logical_or(a1, b1).sum())
        iou = 1.0 if union == 0 else float(np.logical_and(a1, b1).sum() / union)
        print(
            f"{stem}.png: Python(OpenCV {cv2.__version__} + ORT 关键点) vs C++ 内核 → "
            f"MAE={d.mean():.4f} max={int(d.max())} 差异像素={float((d.max(axis=2) > 0).mean() * 100):.3f}% "
            f"IoU={iou:.4f} 逐位相同={bool(d.max() == 0 and np.array_equal(out.highlight_mask, ref_mask))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
