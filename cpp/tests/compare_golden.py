# -*- coding: utf-8 -*-
"""黄金对照：比较 C++ 版与 Python 版的去高光结果。

对每个样本比较：
- result：max abs diff / mean abs diff / 不同像素比例
- hard/soft/skin/protect/treatable mask：IoU（soft 按 >0 二值化再算 IoU，另报 max diff）

用法：
  .venv/bin/python cpp/tests/compare_golden.py --golden cpp/golden --cpp-bin cpp/build/facehi_cpp \
      --names 1 2 3 5 9 13 [--use-golden-landmarks]
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


def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def mask_iou(a: np.ndarray, b: np.ndarray, thr: int = 0) -> float:
    aa = a > thr
    bb = b > thr
    union = int((aa | bb).sum())
    if union == 0:
        return 1.0
    return float((aa & bb).sum() / union)


def compare_one(name: str, golden_dir: Path, cpp_bin: Path, use_golden_landmarks: bool) -> dict:
    input_png = golden_dir / f"{name}_input.png"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        out_png = tmp_dir / f"{name}_cpp.png"
        cmd = [str(cpp_bin), "-i", str(input_png), "-o", str(out_png),
               "--dump-dir", str(tmp_dir), "--repo-root", str(ROOT)]
        if use_golden_landmarks:
            cmd += ["--landmarks", str(golden_dir / f"{name}_landmarks.txt")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return {"name": name, "ok": False, "error": r.stderr.strip()[-500:]}

        res = {"name": name, "ok": True}
        g_result = imread(golden_dir / f"{name}_result.png")
        c_result = imread(out_png)
        diff = np.abs(g_result.astype(np.int32) - c_result.astype(np.int32))
        res["result_max_abs_diff"] = int(diff.max())
        res["result_mean_abs_diff"] = float(diff.mean())
        res["result_diff_pixel_ratio"] = float((diff.max(axis=2) > 0).mean())

        for mask_name in ["hard", "soft", "skin", "protect", "treatable"]:
            gp = golden_dir / f"{name}_{mask_name}.png"
            # CLI 以输入文件名 stem（如 1_input）命名转储文件
            cp = tmp_dir / f"{name}_input_{mask_name}.png"
            if not gp.exists() or not cp.exists():
                continue
            g = imread(gp)
            c = imread(cp)
            res[f"{mask_name}_iou"] = round(mask_iou(g, c), 6)
            res[f"{mask_name}_max_diff"] = int(np.abs(g.astype(np.int32) - c.astype(np.int32)).max())
        return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "cpp" / "golden"))
    ap.add_argument("--cpp-bin", default=str(ROOT / "cpp" / "build" / "facehi_cpp"))
    ap.add_argument("--names", nargs="+", default=["1", "2", "3", "5", "9", "13"])
    ap.add_argument("--use-golden-landmarks", action="store_true",
                    help="注入 Python 导出的关键点，只对比 CV 流水线（排除推理引擎差异）")
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    cpp_bin = Path(args.cpp_bin)
    if not cpp_bin.exists():
        print(f"[错误] 找不到 C++ 可执行文件：{cpp_bin}", file=sys.stderr)
        return 2

    all_ok = True
    for name in args.names:
        res = compare_one(name, golden_dir, cpp_bin, args.use_golden_landmarks)
        if not res.get("ok"):
            all_ok = False
            print(f"== {name}: 运行失败\n{res.get('error')}")
            continue
        print(f"== {name}: result max={res['result_max_abs_diff']} "
              f"mean={res['result_mean_abs_diff']:.6f} "
              f"diff像素比例={res['result_diff_pixel_ratio']:.6f} | "
              f"hard IoU={res.get('hard_iou', 'n/a')} soft IoU={res.get('soft_iou', 'n/a')} "
              f"skin IoU={res.get('skin_iou', 'n/a')} protect IoU={res.get('protect_iou', 'n/a')} "
              f"treatable IoU={res.get('treatable_iou', 'n/a')}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
