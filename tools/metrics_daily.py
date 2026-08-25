"""日常版（configs/onnx_daily.yaml + models/high_removal_daily.onnx）实测指标，写入 tools/metrics_daily.md。

在 data/1.png、15.png、16.png 上真实运行并报告：
  1. 硬掩码内 mean |ΔL|（OpenCV LAB L，0~255）——处理前后的平均亮度变化幅度；
  2. 皮肤区 L>=200 亮斑像素数 处理前→处理后；
  3. 整图 MAE（vs 原图，三通道平均绝对差）；
  4. 一致性验证：session.run(models/high_removal_daily.onnx, 精确内核 libhigh_removal_pyops.so)
     与 process_image(cfg=configs/onnx_daily.yaml) 是否 np.array_equal（result 与 hard_mask）；
  5. 附加：日常版 vs 当前默认流水线（configs/default.yaml / models/high_removal.onnx）是否逐位相同
     ——onnx_daily.yaml 除顶部注释外与 default.yaml 逐字节一致，预期逐位相同。

用法（仓库根目录）：python tools/metrics_daily.py
"""
from __future__ import annotations

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
from highlight_removal.utils import apply_runtime_mode, load_yaml, mask_to_uint8  # noqa: E402

IMAGES = ("1", "15", "16")
DAILY_YAML = ROOT / "configs" / "onnx_daily.yaml"
DAILY_ONNX = ROOT / "models" / "high_removal_daily.onnx"
DEFAULT_YAML = ROOT / "configs" / "default.yaml"
DEFAULT_ONNX = ROOT / "models" / "high_removal.onnx"
EXACT_KERNEL = ROOT / "cpp" / "build" / "libhigh_removal_pyops.so"
REPORT = ROOT / "tools" / "metrics_daily.md"


def build_cfg(yaml_path: Path) -> dict:
    """与精确内核 highlight_removal.exact_kernel._build_config 完全一致的配置构建。"""
    cfg = load_yaml(yaml_path)
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def make_session(model: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(EXACT_KERNEL))
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])


def main() -> int:
    for p in (DAILY_YAML, DAILY_ONNX, DEFAULT_YAML, DEFAULT_ONNX, EXACT_KERNEL):
        if not p.is_file():
            print(f"[错误] 缺少 {p}", file=sys.stderr)
            return 2

    cfg_daily = build_cfg(DAILY_YAML)
    cfg_default = build_cfg(DEFAULT_YAML)
    if not preload_models(cfg_daily):
        return 3

    sess_daily = make_session(DAILY_ONNX)
    sess_default = make_session(DEFAULT_ONNX)

    rows = []
    for stem in IMAGES:
        img = cv2.imread(str(ROOT / "data" / f"{stem}.png"), cv2.IMREAD_COLOR)
        out = process_image(img, cfg_daily)
        assert out.success, f"{stem}.png 处理失败: {out.status}"

        L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        L1 = cv2.cvtColor(out.result_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        hard = out.highlight_mask > 0
        skin = mask_to_uint8(out.regions.masks["skin"]) > 0

        mean_abs_dl = float(np.abs(L1 - L0)[hard].mean()) if hard.any() else 0.0
        mean_l_before = float(L0[hard].mean()) if hard.any() else float("nan")
        mean_l_after = float(L1[hard].mean()) if hard.any() else float("nan")
        bright_before = int(((L0 >= 200) & skin).sum())
        bright_after = int(((L1 >= 200) & skin).sum())
        mae = float(np.abs(out.result_bgr.astype(np.int16) - img.astype(np.int16)).mean())

        feed = {"image": np.ascontiguousarray(img)}
        d_res, d_hard = sess_daily.run(["result", "hard_mask"], feed)
        onnx_equal = bool(np.array_equal(out.result_bgr, d_res) and np.array_equal(out.highlight_mask, d_hard))

        out_def = process_image(img, cfg_default)
        py_default_equal = bool(
            np.array_equal(out.result_bgr, out_def.result_bgr)
            and np.array_equal(out.highlight_mask, out_def.highlight_mask)
        )
        f_res, f_hard = sess_default.run(["result", "hard_mask"], feed)
        onnx_default_equal = bool(np.array_equal(d_res, f_res) and np.array_equal(d_hard, f_hard))

        rows.append({
            "name": f"{stem}.png",
            "hard_px": int(hard.sum()),
            "mean_abs_dl": mean_abs_dl,
            "mean_l_before": mean_l_before,
            "mean_l_after": mean_l_after,
            "bright_before": bright_before,
            "bright_after": bright_after,
            "mae": mae,
            "onnx_equal": onnx_equal,
            "py_default_equal": py_default_equal,
            "onnx_default_equal": onnx_default_equal,
        })
        print(
            f"{stem}.png: 硬掩码 {rows[-1]['hard_px']}px  掩码内|ΔL|={mean_abs_dl:.2f}"
            f"（均L {mean_l_before:.1f}→{mean_l_after:.1f}）  皮肤L≥200 {bright_before}→{bright_after}px  "
            f"MAE={mae:.4f}  ONNX逐位相同={onnx_equal}  与default流水线逐位相同={py_default_equal}  "
            f"与high_removal.onnx逐位相同={onnx_default_equal}"
        )

    # 配置体对比：去掉 onnx_daily.yaml 顶部的注释块后应与 default.yaml 逐字节一致。
    daily_lines = DAILY_YAML.read_text(encoding="utf-8").splitlines(keepends=True)
    body_start = next(i for i, line in enumerate(daily_lines) if line.strip() and not line.startswith("#"))
    body_identical = "".join(daily_lines[body_start:]) == DEFAULT_YAML.read_text(encoding="utf-8")

    import mediapipe
    import platform

    all_onnx_equal = all(r["onnx_equal"] for r in rows)
    all_default_equal = all(r["py_default_equal"] and r["onnx_default_equal"] for r in rows)
    lines = [
        "# 日常版（onnx_daily）实测指标",
        "",
        f"自动生成：`python tools/metrics_daily.py`（{datetime.date.today().isoformat()}）。",
        "",
        "- 配置：`configs/onnx_daily.yaml`（日常版 = 当前默认平衡去高光；除顶部注释外与 "
        "`configs/default.yaml` 逐字节一致，`highlight_detection` / `highlight_removal` 参数完全相同）",
        "- 模型：`models/high_removal_daily.onnx`（`python tools/export_high_removal_onnx.py "
        "--config configs/onnx_daily.yaml --output models/high_removal_daily.onnx`）",
        "- ONNX 侧：`onnxruntime.InferenceSession` + 精确内核 `cpp/build/libhigh_removal_pyops.so`，"
        "一次 `session.run` 得到 result / hard_mask",
        "- Python 参考侧：`highlight_removal.pipeline.process_image`，配置 = onnx_daily.yaml "
        "+ 默认 CPU 运行时 + 关闭可视化（与精确内核对内嵌配置的处理完全一致）",
        "- L = OpenCV LAB 的 L 通道（0~255）；皮肤区 = 流水线肤色掩码 `regions.masks[\"skin\"]`；"
        "MAE = 整图三通道平均绝对差（vs 原图）",
        "",
        "| 图片 | 硬掩码像素 | 掩码内 mean \\|ΔL\\| | 掩码内均 L 前→后 | 皮肤 L≥200 前→后 (px) | MAE vs 原图 | "
        "session.run ≟ process_image | ≟ default 流水线 | ≟ high_removal.onnx |",
        "|------|-----------|------------------|-----------------|----------------------|-------------|"
        "----------------------------|-----------------|--------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['hard_px']} | {r['mean_abs_dl']:.2f} | "
            f"{r['mean_l_before']:.1f}→{r['mean_l_after']:.1f} | {r['bright_before']}→{r['bright_after']} | "
            f"{r['mae']:.4f} | {'✅ np.array_equal' if r['onnx_equal'] else '❌'} | "
            f"{'✅' if r['py_default_equal'] else '❌'} | {'✅' if r['onnx_default_equal'] else '❌'} |"
        )
    lines += [
        "",
        f"- **一致性结论**：3/3 图片上 `session.run(models/high_removal_daily.onnx)`（精确内核 "
        f"libhigh_removal_pyops.so）与 `process_image(cfg=configs/onnx_daily.yaml)` np.array_equal（result 与 "
        f"hard_mask 均逐位相同）：**{'是' if all_onnx_equal else '否'}**。",
        f"- **日常版 ≟ 当前默认**：onnx_daily.yaml 与 default.yaml 的配置体（注释外）逐字节一致：**"
        f"{'是' if body_identical else '否'}**；3/3 图片上日常版输出与 default.yaml 流水线及现有 "
        f"models/high_removal.onnx（同一精确内核）逐位相同：**{'是' if all_default_equal else '否'}**"
        "——日常版 = 当前生产平衡效果的具名版本，仅文件名/内嵌配置名不同。",
        "",
        f"测试环境：Python {platform.python_version()}，onnxruntime {ort.__version__}，"
        f"opencv-contrib-python {cv2.__version__}，mediapipe {mediapipe.__version__}，"
        "精确内核 g++ 13.3（Ubuntu 24.04）+ ONNX Runtime 1.22.0 头文件构建。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
