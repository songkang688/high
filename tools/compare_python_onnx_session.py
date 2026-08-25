"""Python 原始流程 vs 单 ONNX 会话（models/high_removal.onnx + libhigh_removal_ops.so）输出对拍。

对每张图：
  1. 用与 cli_process.py「常用模式」相同的配置跑 Python process_image（MediaPipe + OpenCV 原始链路）；
  2. 在同一进程内用 onnxruntime 加载单 ONNX 模型（register_custom_ops_library）跑 session.run；
  3. 统计结果图 MAE / PSNR / 最大差 / 差异像素占比，以及硬掩码 IoU。

结果打印到终端，并把「单 ONNX 会话 vs Python」章节写入 tools/PARITY_RESULTS.md
（保留文件中已有的其它章节；本章节用标记幂等替换）。

用法：
  python tools/compare_python_onnx_session.py [--images data/1.png ...]
  python tools/compare_python_onnx_session.py --ops cpp/build/libhigh_removal_ops.so
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from cli_process import load_cli_config, preload_models  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402

SECTION_BEGIN = "<!-- single-onnx-session-begin -->"
SECTION_END = "<!-- single-onnx-session-end -->"


def metrics(py_img: np.ndarray, onnx_img: np.ndarray) -> dict:
    d = np.abs(py_img.astype(np.int16) - onnx_img.astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return {
        "mae": float(d.mean()),
        "max": int(d.max()),
        "diff_pixel_pct": float((d.max(axis=2) > 0).mean() * 100.0),
        "psnr": float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a > 0
    b1 = b > 0
    union = int(np.logical_or(a1, b1).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a1, b1).sum() / union)


def build_section(rows: list, summary: str | None) -> str:
    lines = [
        SECTION_BEGIN,
        "## 单 ONNX 会话 vs Python（single ONNX session vs Python）",
        "",
        "自动生成：`python tools/compare_python_onnx_session.py`。",
        "",
        "- Python 侧：`cli_process.py` 常用模式（= configs/default.yaml 全默认路径，MediaPipe + OpenCV 原始链路）",
        "- ONNX 侧：`onnxruntime.InferenceSession(\"models/high_removal.onnx\")` + "
        "`register_custom_ops_library(libhigh_removal_ops.so)`，一次 `session.run` 得到 result / hard_mask",
        "- 单 ONNX 会话输出与旧 `cpp/build/high_onnx` CLI 输出**逐位相同**（自定义算子内核调用同一 C++ 流水线），"
        "因此与 Python 的差异与上表同源：推理引擎浮点尾差与 OpenCV 版本差异，非位级 100%，属如实量化",
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
    if summary:
        lines += ["", summary]
    import platform

    import mediapipe

    lines += [
        "",
        "测试环境：Python "
        f"{platform.python_version()}，onnxruntime {ort.__version__}，opencv-python {cv2.__version__}，"
        f"mediapipe {mediapipe.__version__}；自定义算子库 g++ 13.3（Ubuntu 24.04）+ OpenCV 4.6.0（apt）+ "
        "ONNX Runtime 1.22.0 头文件（运行时 OrtApi 由宿主提供）。",
        SECTION_END,
    ]
    return "\n".join(lines)


def write_report(report_path: Path, section: str) -> None:
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8")
        if SECTION_BEGIN in text and SECTION_END in text:
            head, rest = text.split(SECTION_BEGIN, 1)
            _, tail = rest.split(SECTION_END, 1)
            text = head.rstrip() + "\n\n" + section + tail
        else:
            text = text.rstrip() + "\n\n" + section + "\n"
    else:
        text = "# 输出对拍结果\n\n" + section + "\n"
    report_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "high_removal.onnx"))
    parser.add_argument("--ops", default=str(ROOT / "cpp" / "build" / "libhigh_removal_ops.so"))
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--report", default=str(ROOT / "tools" / "PARITY_RESULTS.md"))
    args = parser.parse_args()

    if not Path(args.ops).is_file():
        print(f"[错误] 找不到自定义算子库 {args.ops}，请先编译（见 cpp/README.md）", file=sys.stderr)
        return 2
    if not Path(args.model).is_file():
        print(f"[错误] 找不到 {args.model}，请先运行 tools/export_high_removal_onnx.py", file=sys.stderr)
        return 2
    images = [
        Path(p)
        for p in (args.images or sorted((ROOT / "data").glob("*.png"), key=lambda p: (len(p.stem), p.stem)))
    ]

    cfg = load_cli_config("常用模式")
    if not preload_models(cfg):
        return 3

    so = ort.SessionOptions()
    so.register_custom_ops_library(args.ops)
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])

    rows = []
    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[跳过] 读不到 {image_path}")
            continue
        py_out = process_image(img, cfg)
        if not py_out.success:
            print(f"[跳过] Python 处理失败 {image_path}: {py_out.status}")
            continue

        try:
            onnx_result, onnx_hard = sess.run(["result", "hard_mask"], {"image": np.ascontiguousarray(img)})
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] ONNX 会话处理失败 {image_path}: {exc}")
            rows.append((image_path.name, None))
            continue

        m = metrics(py_out.result_bgr, onnx_result)
        m["hard_iou"] = mask_iou(py_out.highlight_mask, onnx_hard)
        rows.append((image_path.name, m))
        print(
            f"{image_path.name}: MAE={m['mae']:.4f}  max={m['max']}  "
            f"差异像素={m['diff_pixel_pct']:.3f}%  PSNR={m['psnr']:.2f}dB  硬掩码IoU={m['hard_iou']:.4f}"
        )

    ok_rows = [(name, m) for name, m in rows if m is not None]
    summary = None
    if ok_rows:
        avg_mae = np.mean([m["mae"] for _, m in ok_rows])
        avg_psnr = np.mean([m["psnr"] for _, m in ok_rows if np.isfinite(m["psnr"])])
        avg_iou = np.mean([m["hard_iou"] for _, m in ok_rows])
        worst_max = max(m["max"] for _, m in ok_rows)
        summary = (
            f"**汇总（{len(ok_rows)} 张）**：平均 MAE = {avg_mae:.4f}，平均 PSNR = {avg_psnr:.2f} dB，"
            f"平均硬掩码 IoU = {avg_iou:.4f}，全部图片最大像素差 = {worst_max}/255。"
        )
        print("\n" + summary)

    write_report(Path(args.report), build_section(rows, summary))
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
