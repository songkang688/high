# -*- coding: utf-8 -*-
"""黄金对照：单一 facehi.onnx（ORT 自定义算子）vs Python 版 vs C++ 直调。

对每个样本比较三条路径：
1. session.Run(facehi.onnx) 完整链路（内置 ONNX 关键点） vs Python 黄金结果
2. session.Run(facehi.onnx) 注入 Python 黄金关键点        vs Python 黄金结果
   —— 隔离推理引擎差异，只验证 CV 流水线（目标 max diff = 0）
3. session.Run(facehi.onnx) vs cpp/build/facehi_cpp 直调 process_image
   —— 验证 ONNX 封装不引入任何新误差（目标位级一致）

用法：
  .venv/bin/python onnx/tests/compare_facehi_onnx.py --names 1 2 ... 17
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from facehi_onnx import FacehiOnnx  # noqa: E402


def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_landmarks(path: Path) -> np.ndarray:
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()] for l in lines[1:n + 1]],
                    dtype=np.float32)


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    dmax = int(d.max())
    dmean = float(d.mean())
    ratio = float((d.max(axis=2) > 0).mean()) if d.ndim == 3 else float((d > 0).mean())
    return dmax, dmean, ratio


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    union = int((aa | bb).sum())
    return 1.0 if union == 0 else float((aa & bb).sum() / union)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "cpp" / "golden"))
    ap.add_argument("--cpp-bin", default=str(ROOT / "cpp" / "build" / "facehi_cpp"))
    ap.add_argument("--names", nargs="+",
                    default=[str(i) for i in range(1, 18)])
    ap.add_argument("--skip-cpp", action="store_true")
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    sess = FacehiOnnx()
    print(f"[信息] model={sess.model_path} ops_lib={sess.ops_lib}")

    all_ok = True
    for name in args.names:
        try:
            image = imread(golden_dir / f"{name}_input.png")
            g_result = imread(golden_dir / f"{name}_result.png")
            g_hard = imread(golden_dir / f"{name}_hard.png")

            # 路径 1：完整链路
            r_full, m_full = sess.run(image)
            f_max, f_mean, f_ratio = diff_stats(g_result, r_full)
            f_iou = mask_iou(g_hard, m_full)

            # 路径 2：注入黄金关键点
            lm = load_landmarks(golden_dir / f"{name}_landmarks.txt")
            r_inj, m_inj = sess.run(image, landmarks=lm)
            i_max, i_mean, i_ratio = diff_stats(g_result, r_inj)
            i_iou = mask_iou(g_hard, m_inj)

            line = (f"== {name}: [完整链路] result max={f_max} mean={f_mean:.6f} "
                    f"diff像素比例={f_ratio:.6f} hard IoU={f_iou:.6f} | "
                    f"[注入关键点] result max={i_max} mean={i_mean:.6f} "
                    f"diff像素比例={i_ratio:.6f} hard IoU={i_iou:.6f}")

            # 路径 3：vs C++ 直调
            if not args.skip_cpp:
                with tempfile.TemporaryDirectory() as tmp:
                    out_png = Path(tmp) / "cpp.png"
                    r = subprocess.run(
                        [args.cpp_bin, "-i", str(golden_dir / f"{name}_input.png"),
                         "-o", str(out_png), "--repo-root", str(ROOT)],
                        capture_output=True, text=True)
                    if r.returncode != 0:
                        line += " | [vs C++直调] 运行失败"
                        all_ok = False
                    else:
                        c_result = imread(out_png)
                        c_max, _, _ = diff_stats(c_result, r_full)
                        line += f" | [vs C++直调] max={c_max}"
                        if c_max != 0:
                            all_ok = False
            print(line, flush=True)
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"== {name}: 失败 {e}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
