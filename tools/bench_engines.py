"""三引擎速度对比：原始 Python 流水线 vs 单 ONNX 会话（精确内核 / C++ 快速内核）。

引擎（与 app_studio.py 完全同一配置：default.yaml + 默认 CPU 运行时 + 关闭可视化）：
  1. python_orig —— highlight_removal.pipeline.process_image 本体（“以前的 python”；
     app_studio.py 的 Python 引擎调用的就是同一个函数 + 同一份配置，无第二套实现）；
  2. onnx_exact  —— onnxruntime session.run + 精确内核
     （libhigh_removal_pyops.so / high_removal_pyops.dll，Compute 内调用 process_image 本体）；
  3. onnx_cpp    —— onnxruntime session.run + C++ 快速内核
     （libhigh_removal_ops.so / high_removal_ops.dll，完整 C++ 移植）。

方法：对 data/*.png 全量图片，每个引擎先做 1 次预热（进程内模型加载/会话构建不计时），
然后每张图计时 N 次（默认 3 次）process_image / session.run 调用本身的墙钟时间
（不含 imread / imwrite），取每张图的平均值；汇总给出各引擎的 mean / median。

用法：
  python tools/bench_engines.py                       # 全部 data/*.png，每图 3 次
  python tools/bench_engines.py --runs 1              # 每图 1 次（快速粗测）
  python tools/bench_engines.py --images data/1.png data/2.png
结果写入 tools/SPEED_RESULTS.md（--report 可改路径）。
"""
from __future__ import annotations

import argparse
import datetime
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE, bootstrap_before_numpy  # noqa: E402

bootstrap_before_numpy()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import apply_runtime_mode, load_yaml  # noqa: E402
from tools.run_high_onnx import find_kernel_lib  # noqa: E402

MODEL_PATH = ROOT / "models" / "high_removal.onnx"


def studio_config() -> dict:
    """与 app_studio.py 的 studio_config() / 精确内核内嵌配置完全一致。"""
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def make_session(ops_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(ops_path))
    return ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])


def bench_engine(name: str, call, images: list[tuple[Path, np.ndarray]], runs: int) -> dict[str, list[float]]:
    """call(img) 计时 runs 次/图；预热 1 次（第一张图，不计时）。返回 {图片名: [毫秒...]} 。"""
    print(f"\n== {name}：预热 1 次 + 每图 {runs} 次计时 ==")
    call(images[0][1])  # 预热：模型加载 / 会话与内核初始化不计入
    per_image: dict[str, list[float]] = {}
    for path, img in images:
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            call(img)
            samples.append((time.perf_counter() - t0) * 1000.0)
        per_image[path.name] = samples
        print(f"  {path.name}: {'/'.join(f'{s:.0f}' for s in samples)} ms  (avg {statistics.mean(samples):.0f})")
    return per_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="每张图每个引擎的计时次数（默认 3）")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--report", default=str(ROOT / "tools" / "SPEED_RESULTS.md"))
    args = parser.parse_args()

    image_paths = [
        Path(p)
        for p in (args.images or sorted((ROOT / "data").glob("*.png"), key=lambda p: (len(p.stem), p.stem)))
    ]
    images = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[跳过] 读不到 {p}", file=sys.stderr)
            continue
        images.append((p, np.ascontiguousarray(img)))
    if not images:
        print("[错误] 没有可用图片", file=sys.stderr)
        return 2

    cfg = studio_config()
    exact_lib = find_kernel_lib("exact")
    cpp_lib = find_kernel_lib("cpp")
    for lib in (exact_lib, cpp_lib):
        if not lib.is_file():
            print(f"[错误] 找不到算子库 {lib}，请先按 cpp/README.md 构建", file=sys.stderr)
            return 2

    sess_exact = make_session(exact_lib)
    sess_cpp = make_session(cpp_lib)

    engines = {
        "python_orig": lambda img: process_image(img, cfg),
        "onnx_exact": lambda img: sess_exact.run(["result", "hard_mask"], {"image": img}),
        "onnx_cpp": lambda img: sess_cpp.run(["result", "hard_mask"], {"image": img}),
    }
    results = {name: bench_engine(name, call, images, args.runs) for name, call in engines.items()}

    # 汇总：每张图取 runs 次平均，再对图片求 mean / median。
    names = [p.name for p, _ in images]
    avg = {e: {n: statistics.mean(results[e][n]) for n in names} for e in engines}
    summary = {
        e: {
            "mean": statistics.mean(avg[e].values()),
            "median": statistics.median(avg[e].values()),
        }
        for e in engines
    }
    ratio_exact = summary["onnx_exact"]["mean"] / summary["python_orig"]["mean"]
    ratio_cpp = summary["onnx_cpp"]["mean"] / summary["python_orig"]["mean"]

    print("\n== 汇总（每图取平均后跨图统计，单位 ms）==")
    for e in engines:
        print(f"  {e:12s} mean={summary[e]['mean']:8.1f}  median={summary[e]['median']:8.1f}")
    print(f"  onnx_exact / python_orig = {ratio_exact:.3f}")
    print(f"  onnx_cpp   / python_orig = {ratio_cpp:.3f}")

    # ---- 写报告 ----
    import mediapipe

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    lines = [
        "# 引擎速度对比（SPEED_RESULTS）",
        "",
        f"自动生成：`python tools/bench_engines.py --runs {args.runs}`（{datetime.date.today().isoformat()}）。",
        "",
        f"- 图片：`data/*.png` 全量 {len(images)} 张；每个引擎预热 1 次后，每张图计时 {args.runs} 次取平均；",
        "- 计时范围：`process_image(img, cfg)` / `session.run(...)` 调用本身的墙钟时间（不含 imread/imwrite）；",
        "- 配置：三个引擎完全一致 —— `configs/default.yaml` + 默认 CPU 运行时（`apply_runtime_mode`）+ 关闭可视化，"
        "即 `app_studio.py` 工作室 Python 引擎的同一份配置（studio 引擎与 python_orig 调用同一个 "
        "`process_image`，无独立实现，故不单列）；",
        f"- 环境：{cpu}，{platform.system()} {platform.machine()}，Python {platform.python_version()}，"
        f"onnxruntime {ort.__version__}，opencv {cv2.__version__}，mediapipe {mediapipe.__version__}；",
        f"- 算子库：精确内核 `{exact_lib.name}`，C++ 内核 `{cpp_lib.name}`（cpp/ 构建，见 cpp/README.md）；",
        "- 宿主 onnxruntime 版本：精确内核 pip ≥ 1.22 即可；C++ 快速内核在 Python 宿主下需 pip ≥ 1.23"
        "（1.22.x 存在嵌套建会话的 LoggingManager 冲突，见 cpp/README.md）。",
        "",
        "| 图片 | python_orig (ms) | onnx_exact (ms) | onnx_cpp (ms) | exact/python | cpp/python |",
        "|------|-----------------|-----------------|---------------|--------------|------------|",
    ]
    for n in names:
        pv, ev, cv_ = avg["python_orig"][n], avg["onnx_exact"][n], avg["onnx_cpp"][n]
        lines.append(f"| {n} | {pv:.0f} | {ev:.0f} | {cv_:.0f} | {ev / pv:.3f} | {cv_ / pv:.3f} |")
    lines += [
        "",
        "| 引擎 | mean (ms) | median (ms) | 相对 python_orig（mean） |",
        "|------|-----------|-------------|--------------------------|",
        f"| python_orig（原始 Python `process_image`，= studio Python 引擎） "
        f"| {summary['python_orig']['mean']:.0f} | {summary['python_orig']['median']:.0f} | 1.000 |",
        f"| onnx_exact（session.run + 精确内核） "
        f"| {summary['onnx_exact']['mean']:.0f} | {summary['onnx_exact']['median']:.0f} | {ratio_exact:.3f} |",
        f"| onnx_cpp（session.run + C++ 快速内核） "
        f"| {summary['onnx_cpp']['mean']:.0f} | {summary['onnx_cpp']['median']:.0f} | {ratio_cpp:.3f} |",
        "",
        "## 结论",
        "",
    ]
    exact_pct = (ratio_exact - 1.0) * 100.0
    if abs(exact_pct) <= 20.0:
        lines.append(
            f"- **ONNX 精确内核与原始 Python 速度相当**：mean 相差 {exact_pct:+.1f}%（预期内 —— "
            "精确内核就是原始 Python 流水线本体 + ORT 张量进出拷贝的少量开销，阈值 ±20%）。"
        )
    else:
        lines.append(
            f"- **注意**：ONNX 精确内核与原始 Python 的 mean 相差 {exact_pct:+.1f}%，超出 ±20% 预期，"
            "请检查测量环境（负载/降频）后复测。"
        )
    lines.append(
        f"- C++ 快速内核 mean 为 python_orig 的 {ratio_cpp:.3f} 倍"
        + ("（更快）。" if ratio_cpp < 1.0 else "（相近或略慢）。")
    )
    lines += [
        "",
        "> 以上均为本 Linux 机器实测。Windows DLL 仅做了功能与逐位一致性验证"
        "（见 `.github/workflows/windows-dll.yml` 冒烟测试），未在 Windows 上做速度测量。",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
