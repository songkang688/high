# -*- coding: utf-8 -*-
"""git-high2 三引擎平均用时基准（data/1.png…17.png 全量）。

引擎定义（与 bench/BENCH_REPORT.md 一致）：
  - python_orig  ：仓库 cli_process.load_cli_config("常用模式")
                   + highlight_removal.pipeline.process_image（MediaPipe FaceLandmarker）
  - python_studio：git-high2 前端「Python 版」页签（app_git_high2._run_python）——
                   与 python_orig 是同一 load_cli_config + process_image 调用，
                   不重复实现第三条链路，报告中说明即可。
  - onnx_python  ：git-high2/facehi_onnx.py 的 FacehiOnnx（onnxruntime +
                   register_custom_ops_library + models/facehi.onnx），
                   同一进程复用 session。

协议（公平性要点）：
  1. 每个引擎独立子进程运行，先用 data/1.png warmup 1 张（含模型/会话加载，
     计为冷启动），随后对 17 张逐张计时（只计处理调用，解码在计时外）。
  2. 线程与「常用模式」一致：python_orig 走 bootstrap_before_numpy()（CPU 狂暴
     = 全部逻辑核）+ load_cli_config 内 cv2.setNumThreads；onnx_python 用
     facehi_onnx.FacehiOnnx 文档默认 intra_op_threads=0（ORT 自选）。
  3. 可视化关闭（load_cli_config 已固定 enable_visualization=False）。
  4. 每引擎把 data/1.png 的 result/mask 存成 npy，驱动进程做 MAE / max /
     差异像素占比 / IoU 正确性抽检。

用法：
  python git-high2/bench/bench_runtime.py --all            # 跑全部引擎 + 汇总
  python git-high2/bench/bench_runtime.py --engine onnx_python --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # git-high2/bench
_GH2 = _HERE.parent                              # git-high2
_REPO = _GH2.parent                              # 仓库根
DATA_DIR = _REPO / "data"
IMAGES = [DATA_DIR / f"{i}.png" for i in range(1, 18)]
WARMUP_IMAGE = DATA_DIR / "1.png"
ENGINES = ("python_orig", "onnx_python")


def _decode(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr


def run_engine(engine: str, out_dir: Path) -> dict:
    """在当前进程内跑一个引擎；应由 --all 驱动的独立子进程调用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine == "python_orig":
        sys.path.insert(0, str(_REPO))
        # cli_process 在 import 时执行 bootstrap_before_numpy()（CPU 狂暴线程数）
        t0 = time.perf_counter()
        from cli_process import load_cli_config, preload_models
        from highlight_removal.pipeline import process_image

        cfg = load_cli_config("常用模式")   # 内部 cv2.setNumThreads(全部逻辑核)
        if not preload_models(cfg):
            raise RuntimeError("FaceLandmarker 预加载失败")
        setup_s = time.perf_counter() - t0

        def process(bgr):
            out = process_image(bgr, cfg)
            if not out.success:
                raise RuntimeError(f"pipeline 失败: {out.status}")
            return out.result_bgr, out.highlight_mask

    elif engine == "onnx_python":
        sys.path.insert(0, str(_GH2))
        t0 = time.perf_counter()
        from facehi_onnx import FacehiOnnx

        sess = FacehiOnnx()                 # 文档默认：intra_op_threads=0（ORT 自选）
        setup_s = time.perf_counter() - t0

        def process(bgr):
            return sess.run(bgr)            # 同一进程复用 session
    else:
        raise ValueError(f"未知引擎: {engine}")

    import cv2
    import numpy as np

    # 冷启动：模型/会话加载 + 首张（data/1.png）处理
    warm_bgr = _decode(WARMUP_IMAGE)
    t0 = time.perf_counter()
    process(warm_bgr)
    warmup_s = time.perf_counter() - t0

    rows = []
    for img_path in IMAGES:
        bgr = _decode(img_path)
        t0 = time.perf_counter()
        result, mask = process(bgr)
        dt = time.perf_counter() - t0
        rows.append({
            "engine": engine,
            "image": img_path.name,
            "height": int(bgr.shape[0]),
            "width": int(bgr.shape[1]),
            "seconds": dt,
        })
        if img_path.name == "1.png":
            np.save(out_dir / f"{engine}_1_result.npy", result)
            np.save(out_dir / f"{engine}_1_mask.npy", np.squeeze(mask))

    times = [r["seconds"] for r in rows]
    times_sorted = sorted(times)
    p95_idx = max(0, min(len(times_sorted) - 1, round(0.95 * (len(times_sorted) + 1)) - 1))
    summary = {
        "engine": engine,
        "setup_seconds": setup_s,
        "warmup_first_image_seconds": warmup_s,
        "cold_start_seconds": setup_s + warmup_s,
        "warm_mean_seconds": statistics.fmean(times),
        "warm_median_seconds": statistics.median(times),
        "warm_p95_seconds": times_sorted[p95_idx],
        "warm_min_seconds": min(times),
        "warm_max_seconds": max(times),
        "images": len(rows),
        "cv2_threads": int(cv2.getNumThreads()),
        "rows": rows,
    }
    (out_dir / f"{engine}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _accuracy_check(out_dir: Path) -> dict:
    import numpy as np

    py_res = np.load(out_dir / "python_orig_1_result.npy").astype(np.int16)
    ox_res = np.load(out_dir / "onnx_python_1_result.npy").astype(np.int16)
    py_mask = np.load(out_dir / "python_orig_1_mask.npy") > 0
    ox_mask = np.load(out_dir / "onnx_python_1_mask.npy") > 0
    diff = np.abs(py_res - ox_res)
    inter = np.logical_and(py_mask, ox_mask).sum()
    union = np.logical_or(py_mask, ox_mask).sum()
    return {
        "image": "1.png",
        "mae": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "diff_pixel_ratio": float((diff.max(axis=2) > 0).mean()),
        "mask_iou": float(inter / union) if union else 1.0,
    }


def run_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for engine in ENGINES:
        print(f"[bench] 运行引擎 {engine} …", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--engine", engine, "--out-dir", str(out_dir)],
            check=True)

    summaries = {e: json.loads((out_dir / f"{e}.json").read_text(encoding="utf-8"))
                 for e in ENGINES}
    acc = _accuracy_check(out_dir)

    csv_path = out_dir / "bench_raw.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("engine,image,height,width,seconds\n")
        for e in ENGINES:
            for r in summaries[e]["rows"]:
                f.write(f"{r['engine']},{r['image']},{r['height']},{r['width']},{r['seconds']:.6f}\n")

    combined = {
        "env": _env_info(),
        "accuracy_1png_onnx_vs_python": acc,
        "engines": {e: {k: v for k, v in s.items() if k != "rows"}
                    for e, s in summaries.items()},
    }
    (out_dir / "bench_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


def _env_info() -> dict:
    import cv2
    import numpy as np
    import onnxruntime as ort

    info = {
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import mediapipe as mp

        info["mediapipe"] = mp.__version__
    except Exception:
        info["mediapipe"] = "n/a"
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=ENGINES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=str(_HERE / "out"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.all:
        run_all(out_dir)
    elif args.engine:
        s = run_engine(args.engine, out_dir)
        print(json.dumps({k: v for k, v in s.items() if k != "rows"},
                         ensure_ascii=False, indent=2))
    else:
        ap.error("需要 --all 或 --engine")


if __name__ == "__main__":
    main()
