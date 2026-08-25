# -*- coding: utf-8 -*-
"""导出 Python 流水线的黄金对照数据，供 C++ 版逐像素对齐验证。

对每张输入图片导出（与 cli_process.py「常用模式」完全一致的配置）：
- <名字>_input.png        原图（解码后 BGR，再编码，保证两边读到同一像素）
- <名字>_result.png       Python 去高光结果
- <名字>_hard.png         最终硬 mask（highlight_mask）
- <名字>_soft.png         最终软 mask
- <名字>_landmarks.txt    人脸关键点（原图坐标系，478 行，每行 x y z）
- <名字>_skin.png         全分辨率皮肤 mask
- <名字>_protect.png      全分辨率保护 mask
- <名字>_treatable.png    可去高光皮肤区
- <名字>_metrics.json     质量审计指标与警告

用法：
  .venv/bin/python cpp/tests/export_golden.py data/1.png data/2.png ... -o cpp/golden
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy

bootstrap_before_numpy()

import cv2
import numpy as np

from cli_process import load_cli_config, preload_models, read_image_bgr
from highlight_removal.pipeline import process_image
from highlight_removal.utils import mask_to_uint8, write_image


def export_one(image_path: Path, out_dir: Path, cfg: dict) -> dict:
    name = image_path.stem
    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        return {"name": name, "ok": False, "error": "图片读取失败"}
    out = process_image(image_bgr, cfg)
    if not out.success:
        return {"name": name, "ok": False, "error": out.status}

    write_image(out_dir / f"{name}_input.png", image_bgr)
    write_image(out_dir / f"{name}_result.png", out.result_bgr)
    write_image(out_dir / f"{name}_hard.png", mask_to_uint8(out.highlight_mask))
    write_image(out_dir / f"{name}_soft.png", mask_to_uint8(out.soft_mask))
    if out.regions is not None:
        write_image(out_dir / f"{name}_skin.png", mask_to_uint8(out.regions.masks["skin"]))
        write_image(out_dir / f"{name}_protect.png", mask_to_uint8(out.regions.masks["protect"]))
        treat = out.regions.masks.get("treatable_skin")
        if treat is not None:
            write_image(out_dir / f"{name}_treatable.png", mask_to_uint8(treat))

    lm = out.face.landmarks
    with (out_dir / f"{name}_landmarks.txt").open("w", encoding="utf-8") as f:
        f.write(f"# detector={out.face.detector} confidence={out.face.confidence:.6f}\n")
        f.write(f"{len(lm)}\n")
        for p in lm:
            f.write(f"{float(p[0]):.8f} {float(p[1]):.8f} {float(p[2]):.8f}\n")

    payload = {
        "name": name,
        "ok": True,
        "detector": out.face.detector,
        "warnings": out.warnings,
        "metrics": out.metrics,
        "hard_pixels": int((mask_to_uint8(out.highlight_mask) > 0).sum()),
        "image_shape": list(image_bgr.shape),
    }
    with (out_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=str(ROOT / "cpp" / "golden"))
    ap.add_argument("--mode", default="常用模式")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cli_config(args.mode)
    if not preload_models(cfg):
        return 3

    import mediapipe, platform

    versions = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "mediapipe": mediapipe.__version__,
        "numpy": np.__version__,
        "mode": args.mode,
    }
    with (out_dir / "versions.json").open("w", encoding="utf-8") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)

    ok = 0
    for raw in args.images:
        info = export_one(Path(raw), out_dir, cfg)
        status = "OK" if info.get("ok") else f"FAIL: {info.get('error')}"
        print(f"[golden] {info['name']}: {status}")
        if info.get("ok"):
            ok += 1
    print(f"[golden] 完成 {ok}/{len(args.images)}")
    return 0 if ok == len(args.images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
