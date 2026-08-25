"""Python 流程 vs C++ (high_onnx) 输出对拍。

对每张图：
  1. 用与 cli_process.py「常用模式」相同的配置跑 Python process_image；
  2. 调 C++ 可执行 high_onnx（--dump-masks 同时导出硬/软掩码）；
  3. 统计结果图 MAE / PSNR / 最大差 / 差异像素占比，以及硬掩码 IoU。

结果打印到终端并写入 tools/PARITY_RESULTS.md。

用法：
  python tools/compare_python_cpp.py [--cli cpp/build/high_onnx] [--images data/1.png ...]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cli_process import load_cli_config, preload_models  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402

ORT_LIB_HINT = "/opt/ort/onnxruntime-linux-x64-1.22.0/lib"


def run_cpp(cli: Path, image: Path, out_png: Path, mask_prefix: Path) -> None:
    env = dict(os.environ)
    if Path(ORT_LIB_HINT).is_dir():
        env["LD_LIBRARY_PATH"] = ORT_LIB_HINT + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        [
            str(cli),
            "--input", str(image),
            "--output", str(out_png),
            "--config", str(ROOT / "configs" / "default.yaml"),
            "--models", str(ROOT / "models"),
            "--dump-masks", str(mask_prefix),
        ],
        check=True,
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def metrics(py_img: np.ndarray, cpp_img: np.ndarray) -> dict:
    d = np.abs(py_img.astype(np.int16) - cpp_img.astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return {
        "mae": float(d.mean()),
        "max": int(d.max()),
        "diff_pixel_pct": float((d.max(axis=2) > 0).mean() * 100.0),
        "psnr": float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf"),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a > 0
    b1 = b > 0
    union = int(np.logical_or(a1, b1).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a1, b1).sum() / union)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", default=str(ROOT / "cpp" / "build" / "high_onnx"))
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--report", default=str(ROOT / "tools" / "PARITY_RESULTS.md"))
    args = parser.parse_args()

    cli = Path(args.cli)
    if not cli.is_file():
        print(f"[错误] 找不到 C++ 可执行文件 {cli}，请先编译（见 cpp/README.md）", file=sys.stderr)
        return 2
    images = [Path(p) for p in (args.images or sorted((ROOT / "data").glob("*.png"), key=lambda p: (len(p.stem), p.stem)))]

    cfg = load_cli_config("常用模式")
    if not preload_models(cfg):
        return 3

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for image_path in images:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[跳过] 读不到 {image_path}")
                continue
            py_out = process_image(img, cfg)
            if not py_out.success:
                print(f"[跳过] Python 处理失败 {image_path}: {py_out.status}")
                continue

            cpp_png = tmp_dir / f"{image_path.stem}_cpp.png"
            mask_prefix = tmp_dir / image_path.stem
            try:
                run_cpp(cli, image_path, cpp_png, mask_prefix)
            except subprocess.CalledProcessError as exc:
                print(f"[失败] C++ 处理失败 {image_path}: {exc}")
                rows.append((image_path.name, None))
                continue
            cpp_img = cv2.imread(str(cpp_png), cv2.IMREAD_COLOR)
            cpp_hard = cv2.imread(str(mask_prefix) + "_hard_mask.png", cv2.IMREAD_GRAYSCALE)

            m = metrics(py_out.result_bgr, cpp_img)
            m["hard_iou"] = mask_iou(py_out.highlight_mask, cpp_hard)
            rows.append((image_path.name, m))
            print(
                f"{image_path.name}: MAE={m['mae']:.4f}  max={m['max']}  "
                f"差异像素={m['diff_pixel_pct']:.3f}%  PSNR={m['psnr']:.2f}dB  硬掩码IoU={m['hard_iou']:.4f}"
            )

    ok_rows = [(name, m) for name, m in rows if m is not None]
    if ok_rows:
        avg_mae = np.mean([m["mae"] for _, m in ok_rows])
        avg_psnr = np.mean([m["psnr"] for _, m in ok_rows if np.isfinite(m["psnr"])])
        avg_iou = np.mean([m["hard_iou"] for _, m in ok_rows])
        worst_max = max(m["max"] for _, m in ok_rows)
        print(f"\n共 {len(ok_rows)} 张：平均 MAE={avg_mae:.4f}  平均 PSNR={avg_psnr:.2f}dB  平均硬掩码 IoU={avg_iou:.4f}  最大像素差={worst_max}")

    # 写报告
    lines = [
        "# Python vs C++ 输出对拍结果",
        "",
        "自动生成：`python tools/compare_python_cpp.py`。",
        "",
        "- Python 侧：`cli_process.py` 常用模式（= configs/default.yaml 全默认路径）",
        "- C++ 侧：`cpp/build/high_onnx --config configs/default.yaml`",
        "- 指标基于结果图逐像素差（0~255）与高光硬掩码 IoU",
        "",
        "| 图片 | MAE | 最大像素差 | 差异像素占比 | PSNR (dB) | 硬掩码 IoU |",
        "|------|-----|-----------|-------------|-----------|-----------|",
    ]
    for name, m in rows:
        if m is None:
            lines.append(f"| {name} | 处理失败 | - | - | - | - |")
        else:
            psnr = "inf" if not np.isfinite(m["psnr"]) else f"{m['psnr']:.2f}"
            lines.append(
                f"| {name} | {m['mae']:.4f} | {m['max']} | {m['diff_pixel_pct']:.3f}% | {psnr} | {m['hard_iou']:.4f} |"
            )
    if ok_rows:
        lines += [
            "",
            f"**汇总（{len(ok_rows)} 张）**：平均 MAE = {avg_mae:.4f}，平均 PSNR = {avg_psnr:.2f} dB，"
            f"平均硬掩码 IoU = {avg_iou:.4f}，全部图片最大像素差 = {worst_max}/255。",
        ]
    import mediapipe
    import platform
    lines += [
        "",
        "## 测试环境",
        "",
        f"- Python {platform.python_version()}，opencv-python {cv2.__version__}，mediapipe {mediapipe.__version__}",
        "- C++ 侧：g++ 13.3（Ubuntu 24.04），OpenCV 4.6.0（apt libopencv-dev），ONNX Runtime 1.22.0（官方预编译包），yaml-cpp 0.8",
        "- 剩余差异来源：MediaPipe(TFLite/XNNPACK) 与 ONNX Runtime 的浮点尾差导致关键点亚像素级偏移，"
        "经整数栅格化后在掩码边界处放大为个别像素差；以及 Python/C++ 两侧 OpenCV 版本不同。",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
