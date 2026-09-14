# -*- coding: utf-8 -*-
"""对 output-1.22.0 做冒烟：ORT 版本、会话创建、data/1.png 推理。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "output-1.22.0"
sys.path.insert(0, str(OUT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from facehi_onnx import FacehiOnnx  # noqa: E402


def main() -> int:
    print(f"onnxruntime={ort.__version__}")
    print(f"onnx={onnx.__version__}")
    print(f"cv2={cv2.__version__}")
    if not ort.__version__.startswith("1.22"):
        print("ERROR: 需要 onnxruntime 1.22.x", file=sys.stderr)
        return 2
    model = onnx.load(str(OUT / "facehi.onnx"))
    meta = {p.key: p.value for p in model.metadata_props}
    print(f"onnx.ir_version={model.ir_version} producer={model.producer_name}/{model.producer_version}")
    print(f"onnx.metadata={meta}")

    sess = FacehiOnnx()
    print(f"session model={sess.model_path} ops={sess.ops_lib}")

    img_path = ROOT / "data" / "1.png"
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    result, mask = sess.run(img)
    assert result.shape == img.shape and result.dtype == np.uint8
    assert mask.shape[:2] == img.shape[:2] and mask.dtype == np.uint8
    changed = int(np.mean(result != img) * 10000) / 100.0
    print(f"data/1.png -> result{result.shape} mask{mask.shape} changed_pct={changed:.2f} mask_px={int((mask > 0).sum())}")

    # 与旧库产物并存：确认 git-high2/lib 与 models 仍在
    old_so = Path(__file__).resolve().parents[1] / "lib" / "libfacehi_custom_ops.so"
    old_dll = Path(__file__).resolve().parents[1] / "lib" / "facehi_custom_ops.dll"
    old_onnx = Path(__file__).resolve().parents[1] / "models" / "facehi.onnx"
    for p in (old_so, old_dll, old_onnx):
        print(f"kept_original {p.name} exists={p.is_file()} size={p.stat().st_size if p.is_file() else 0}")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
