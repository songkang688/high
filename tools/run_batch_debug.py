"""批量调试脚本：使用完整 pipeline 处理文件夹中的证件照。

示例：
python tools/run_batch_debug.py --input ./samples --output ./batch_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy, DEFAULT_RUNTIME_MODE

bootstrap_before_numpy()

import cv2
import yaml

from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import write_image, apply_runtime_mode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片文件夹")
    parser.add_argument("--output", default="batch_outputs", help="输出文件夹")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="配置文件")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []
    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            records.append({"file": str(p), "success": False, "status": "读取失败"})
            continue
        out = process_image(img, cfg)
        stem = p.stem
        rec = {"file": str(p), "success": out.success, "status": out.status, "warnings": out.warnings, "metrics": out.metrics}
        records.append(rec)
        if out.success:
            write_image(out_dir / f"{stem}_result.png", out.result_bgr)
            write_image(out_dir / f"{stem}_mask.png", out.highlight_mask)
            write_image(out_dir / f"{stem}_diff.png", out.diff_bgr)
            write_image(out_dir / f"{stem}_regions.png", out.face_region_view_bgr)
            write_image(out_dir / f"{stem}_landmarks.png", out.landmark_view_bgr)
        else:
            write_image(out_dir / f"{stem}_failed_input.png", img)
    with open(out_dir / "batch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"完成：{len(records)} 张，输出目录：{out_dir}")


if __name__ == "__main__":
    main()
