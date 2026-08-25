"""Python 原始流程 vs 单 ONNX 会话（models/high_removal.onnx + 自定义算子库）输出对拍。

同一个 ONNX 文件支持两个内核库，分别对拍：
  - 精确内核 libhigh_removal_pyops.so：session.run 在宿主进程内调用原始 Python 流水线
    （highlight_removal.exact_kernel → process_image），目标是与 Python **逐位相同**
    （np.array_equal，MAE=0，最大差=0，IoU=1.0）；
  - C++ 快速内核 libhigh_removal_ops.so：完整 C++ 移植，存在亚像素级浮点尾差
    （来源拆解见 tools/parity_isolate.py 的章节）。

Python 参考侧配置 = models/high_removal.onnx 内嵌的 configs/default.yaml
（+ 默认 CPU 运行时 + 关闭可视化），与 app_studio.py 的 Python 引擎完全一致。
注：CLI「常用模式」与 default.yaml 原始值唯一差异是
highlight_detection.brow_region_max_fraction（0.32 vs 0.28），实测对 17 张样例图
输出无影响（见误差来源隔离章节）。

结果打印到终端，并把「单 ONNX 会话 vs Python」章节写入 tools/PARITY_RESULTS.md
（保留文件中已有的其它章节；本章节用标记幂等替换）。

用法：
  python tools/compare_python_onnx_session.py                 # 两个内核都测
  python tools/compare_python_onnx_session.py --kernel exact  # 只测精确内核
  python tools/compare_python_onnx_session.py --kernel cpp    # 只测 C++ 内核
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy  # noqa: E402

bootstrap_before_numpy()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from cli_process import preload_models  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE  # noqa: E402
from highlight_removal.utils import apply_runtime_mode, load_yaml  # noqa: E402

SECTION_BEGIN = "<!-- single-onnx-session-begin -->"
SECTION_END = "<!-- single-onnx-session-end -->"

KERNELS = {
    "exact": ("精确内核 libhigh_removal_pyops.so", ROOT / "cpp" / "build" / "libhigh_removal_pyops.so"),
    "cpp": ("C++ 快速内核 libhigh_removal_ops.so", ROOT / "cpp" / "build" / "libhigh_removal_ops.so"),
}


def reference_config() -> dict:
    """与 models/high_removal.onnx 内嵌配置等价（= app_studio.py Python 引擎配置）。"""
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


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


def run_kernel(kernel: str, model: str, images, py_outputs) -> tuple[list, str | None]:
    label, ops = KERNELS[kernel]
    if not ops.is_file():
        print(f"[跳过] {label} 未构建（{ops}）", file=sys.stderr)
        return [], None
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(ops))
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])

    rows = []
    for image_path, img, py_out in zip(*py_outputs):
        try:
            onnx_result, onnx_hard = sess.run(["result", "hard_mask"], {"image": np.ascontiguousarray(img)})
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {label} 处理失败 {image_path}: {exc}", file=sys.stderr)
            rows.append((image_path.name, None))
            continue
        m = metrics(py_out.result_bgr, onnx_result)
        m["hard_iou"] = mask_iou(py_out.highlight_mask, onnx_hard)
        m["bit_equal"] = bool(
            np.array_equal(py_out.result_bgr, onnx_result) and np.array_equal(py_out.highlight_mask, onnx_hard)
        )
        rows.append((image_path.name, m))
        print(
            f"[{kernel}] {image_path.name}: 逐位相同={'是' if m['bit_equal'] else '否'}  "
            f"MAE={m['mae']:.4f}  max={m['max']}  差异像素={m['diff_pixel_pct']:.3f}%  "
            f"硬掩码IoU={m['hard_iou']:.4f}"
        )

    ok_rows = [(n, m) for n, m in rows if m is not None]
    summary = None
    if ok_rows:
        n_equal = sum(1 for _, m in ok_rows if m["bit_equal"])
        avg_mae = np.mean([m["mae"] for _, m in ok_rows])
        finite = [m["psnr"] for _, m in ok_rows if np.isfinite(m["psnr"])]
        avg_psnr = f"{np.mean(finite):.2f} dB" if finite else "inf（全部逐位相同）"
        avg_iou = np.mean([m["hard_iou"] for _, m in ok_rows])
        worst_max = max(m["max"] for _, m in ok_rows)
        summary = (
            f"**{label} 汇总（{len(ok_rows)} 张）**：逐位相同（np.array_equal，含掩码）"
            f" **{n_equal}/{len(ok_rows)}**；平均 MAE = {avg_mae:.4f}，平均 PSNR = {avg_psnr}，"
            f"平均硬掩码 IoU = {avg_iou:.4f}，全部图片最大像素差 = {worst_max}/255。"
        )
        print("\n" + summary + "\n")
    return rows, summary


def kernel_table(label: str, rows: list) -> list[str]:
    lines = [
        f"### {label}",
        "",
        "| 图片 | 逐位相同 | MAE | 最大像素差 | 差异像素占比 | PSNR (dB) | 硬掩码 IoU |",
        "|------|---------|-----|-----------|-------------|-----------|-----------|",
    ]
    for name, m in rows:
        if m is None:
            lines.append(f"| {name} | 处理失败 | - | - | - | - | - |")
        else:
            psnr = "inf" if not np.isfinite(m["psnr"]) else f"{m['psnr']:.2f}"
            eq = "✅" if m["bit_equal"] else "❌"
            lines.append(
                f"| {name} | {eq} | {m['mae']:.4f} | {m['max']} | {m['diff_pixel_pct']:.3f}% | {psnr} | {m['hard_iou']:.4f} |"
            )
    return lines


def build_section(results: dict) -> str:
    import platform

    import mediapipe

    date = datetime.date.today().isoformat()
    lines = [
        SECTION_BEGIN,
        f"## 单 ONNX 会话 vs Python（single ONNX session vs Python，{date}）",
        "",
        "自动生成：`python tools/compare_python_onnx_session.py`。",
        "",
        "- Python 侧：`highlight_removal.pipeline.process_image`，配置 = "
        "models/high_removal.onnx 内嵌的 configs/default.yaml（+ 默认 CPU 运行时 + 关闭可视化），"
        "与 `app_studio.py` Python 引擎完全一致",
        "- ONNX 侧：`onnxruntime.InferenceSession(\"models/high_removal.onnx\")` + "
        "`register_custom_ops_library(算子库)`，一次 `session.run` 得到 result / hard_mask",
        "- 同一个 ONNX 文件、两个可互换内核库：**精确内核**（默认，Compute 内调用原始 Python 流水线本体，"
        "目标逐位相同）与 **C++ 快速内核**（完整 C++ 移植，存在亚像素级浮点尾差，来源拆解见下方隔离章节）",
        "",
    ]
    for label, rows, summary in results.values():
        if not rows:
            lines += [f"### {label}", "", "（未构建，跳过）", ""]
            continue
        lines += kernel_table(label, rows)
        if summary:
            lines += ["", summary, ""]
    lines += [
        "测试环境：Python "
        f"{platform.python_version()}，onnxruntime {ort.__version__}，opencv-python {cv2.__version__}，"
        f"mediapipe {mediapipe.__version__}；算子库 g++ 13.3（Ubuntu 24.04），C++ 内核依赖 OpenCV 4.6.0（apt）+ "
        "ONNX Runtime 1.22.0 头文件，精确内核仅依赖 CPython 稳定 ABI（宿主须为 Python 进程）。",
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
    parser.add_argument("--kernel", choices=["exact", "cpp", "both"], default="both")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--report", default=str(ROOT / "tools" / "PARITY_RESULTS.md"))
    args = parser.parse_args()

    if not Path(args.model).is_file():
        print(f"[错误] 找不到 {args.model}，请先运行 tools/export_high_removal_onnx.py", file=sys.stderr)
        return 2
    kernels = ["exact", "cpp"] if args.kernel == "both" else [args.kernel]
    images = [
        Path(p)
        for p in (args.images or sorted((ROOT / "data").glob("*.png"), key=lambda p: (len(p.stem), p.stem)))
    ]

    cfg = reference_config()
    if not preload_models(cfg):
        return 3

    paths, imgs, py_outs = [], [], []
    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[跳过] 读不到 {image_path}", file=sys.stderr)
            continue
        py_out = process_image(img, cfg)
        if not py_out.success:
            print(f"[跳过] Python 处理失败 {image_path}: {py_out.status}", file=sys.stderr)
            continue
        paths.append(image_path)
        imgs.append(img)
        py_outs.append(py_out)

    results = {}
    for kernel in kernels:
        label = KERNELS[kernel][0]
        rows, summary = run_kernel(kernel, args.model, images, (paths, imgs, py_outs))
        results[kernel] = (label, rows, summary)

    write_report(Path(args.report), build_section(results))
    print(f"报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
