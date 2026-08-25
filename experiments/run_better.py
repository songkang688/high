"""用「实验性纹理优先组合」（configs/experimental_better.yaml）处理图片。

等价于常用模式 + 三个实验开关（extreme_core_extra_l=30 /
texture_keep_factor=0.35 / faithful_baseline_exclude_shine=true），
不改动 configs/default.yaml 的任何默认值。

用法：.venv/bin/python experiments/run_better.py -i photo.png [-o out.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

import cv2  # noqa: E402

from cli_process import load_cli_config, preload_models, read_image_bgr  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import load_yaml, write_image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    cfg = load_cli_config("常用模式")
    overlay = load_yaml(ROOT / "configs" / "experimental_better.yaml")
    cfg["highlight_removal"].update(overlay.get("highlight_removal") or {})
    if not preload_models(cfg):
        return 3

    src = Path(args.input)
    img = read_image_bgr(src)
    if img is None:
        print(f"[错误] 读取失败：{src}", file=sys.stderr)
        return 2
    out = process_image(img, cfg)
    if not out.success:
        print(f"[失败] {out.status}", file=sys.stderr)
        return 5
    dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_better{src.suffix}")
    write_image(dst, out.result_bgr)
    print(dst)
    for w in out.warnings:
        print(f"[警告] {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
