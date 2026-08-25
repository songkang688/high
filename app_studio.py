"""证件照去高光 · Python / ONNX 工作室（推荐 UI：引擎切换 + 并排对比）。

三档强度预设（选择哪档，两个引擎就用哪档的同一套参数，保证公平对比）：
  - 强力    ：models/high_removal_strong.onnx + configs/onnx_strong.yaml（去油光最强）；
  - 日常    ：models/high_removal_daily.onnx + configs/onnx_daily.yaml（默认；= 当前
              default.yaml 平衡效果，输出与现有 high_removal.onnx 逐位相同）；
  - 保护细节：models/high_removal_detail.onnx + configs/onnx_detail.yaml（保留更多皮肤纹理）。
  实测对比见 tools/THREE_ONNX_AUDIT.md。

三种引擎模式：
  - Python：highlight_removal.pipeline.process_image（MediaPipe + OpenCV 原始链路，
            配置 = 所选预设的 yaml）；
  - ONNX  ：所选预设的单文件模型 + 自定义算子库（对应 yaml 已内嵌模型，单次 session.run）。
            默认精确内核 libhigh_removal_pyops.so（与 Python 引擎逐位一致，MAE 恒为 0）；
            设 HIGH_ONNX_KERNEL=cpp 切换 C++ 快速内核 libhigh_removal_ops.so（存在亚像素级浮点尾差）；
  - 对比  ：同一张图同时跑两个引擎，并排显示 原图 / Python / ONNX、差异热力图与量化指标。

用法：
  python app_studio.py                                   # http://127.0.0.1:7861
  HIGH_STUDIO_HOST=0.0.0.0 HIGH_STUDIO_PORT=8080 python app_studio.py
  HIGH_ONNX_KERNEL=cpp python app_studio.py              # ONNX 侧改用 C++ 快速内核
  HIGH_OPS_LIB=/path/to/lib.so python app_studio.py      # 直接指定算子库文件

完整调参调试台（全部滑杆与调试视图）仍是 python app.py（端口 7860）。
"""
from __future__ import annotations

import os

from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE, bootstrap_before_numpy

bootstrap_before_numpy()

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import gradio as gr
import numpy as np

from highlight_removal import onnx_session
from highlight_removal.face_detect import preload_face_landmarker
from highlight_removal.pipeline import process_image
from highlight_removal.utils import apply_runtime_mode, load_yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "runtime_outputs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

ENGINE_PYTHON = "Python"
ENGINE_ONNX = "ONNX"
ENGINE_COMPARE = "对比"
ENGINES = [ENGINE_PYTHON, ENGINE_ONNX, ENGINE_COMPARE]

# 三档强度预设：预设名 → (单文件 ONNX 模型, 对应 yaml)。模型内嵌的 config_yaml 与该 yaml
# 逐字节一致（tools/three_onnx_audit.py 实测），因此 Python 引擎读 yaml、ONNX 引擎用模型，
# 精确内核下两个引擎输出逐位相同，对比模式公平。
PRESET_STRONG = "强力"
PRESET_DAILY = "日常"
PRESET_DETAIL = "保护细节"
PRESETS: Dict[str, tuple] = {
    PRESET_STRONG: (ROOT / "models" / "high_removal_strong.onnx", ROOT / "configs" / "onnx_strong.yaml"),
    PRESET_DAILY: (ROOT / "models" / "high_removal_daily.onnx", ROOT / "configs" / "onnx_daily.yaml"),
    PRESET_DETAIL: (ROOT / "models" / "high_removal_detail.onnx", ROOT / "configs" / "onnx_detail.yaml"),
}
DEFAULT_PRESET = PRESET_DAILY

_config_cache: Dict[str, Dict[str, Any]] = {}
_landmarker_ready = False


# --------------------------------------------------------------------------- 配置与引擎

def studio_config(preset: str = DEFAULT_PRESET) -> Dict[str, Any]:
    """所选预设的 yaml + 运行时线程配置。关闭调试可视化（工作室不需要，计时也更公平）。"""
    if preset not in PRESETS:
        preset = DEFAULT_PRESET
    cfg = _config_cache.get(preset)
    if cfg is None:
        cfg = load_yaml(PRESETS[preset][1])
        cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
        cfg.setdefault("pipeline", {})
        cfg["pipeline"]["enable_visualization"] = False
        _config_cache[preset] = cfg
    return cfg


def ensure_landmarker(cfg: Dict[str, Any]) -> None:
    global _landmarker_ready
    if _landmarker_ready:
        return
    fd = deepcopy(cfg.get("face_detection", {}))
    fd["_runtime"] = cfg.get("runtime", {})
    task_rel = fd.get("mediapipe_task_model_path", "models/face_landmarker.task")
    fd["mediapipe_task_model_path"] = str((ROOT / task_rel).resolve())
    if not preload_face_landmarker(fd):
        raise RuntimeError("人脸模型加载失败：请确认 models/face_landmarker.task 存在且 mediapipe 可用。")
    _landmarker_ready = True


def run_python_engine(image_bgr: np.ndarray, preset: str = DEFAULT_PRESET) -> Dict[str, Any]:
    cfg = studio_config(preset)
    ensure_landmarker(cfg)
    t0 = time.perf_counter()
    out = process_image(image_bgr, cfg)
    elapsed = time.perf_counter() - t0
    if not out.success:
        raise RuntimeError(f"Python 引擎处理失败：{out.status}")
    return {
        "result": out.result_bgr,
        "hard_mask": out.highlight_mask,
        "elapsed": elapsed,
        "warnings": list(out.warnings),
    }


def run_onnx_engine(image_bgr: np.ndarray, preset: str = DEFAULT_PRESET) -> Dict[str, Any]:
    model_path = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])[0]
    session = onnx_session.get_session(model_path=model_path)  # 惰性单例，会话构建不计入耗时
    t0 = time.perf_counter()
    result, hard_mask = onnx_session.remove_highlight(image_bgr, session)
    elapsed = time.perf_counter() - t0
    return {"result": result, "hard_mask": hard_mask, "elapsed": elapsed, "warnings": []}


# --------------------------------------------------------------------------- 指标与可视化

def diff_metrics(a_bgr: np.ndarray, b_bgr: np.ndarray) -> Dict[str, float]:
    d = np.abs(a_bgr.astype(np.int16) - b_bgr.astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return {
        "mae": float(d.mean()),
        "max": int(d.max()),
        "diff_pixel_pct": float((d.max(axis=2) > 0).mean() * 100.0),
        "psnr": float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a1, b1 = a > 0, b > 0
    union = int(np.logical_or(a1, b1).sum())
    return 1.0 if union == 0 else float(np.logical_and(a1, b1).sum() / union)


def diff_heatmap_rgb(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    """逐像素 abs-diff 热力图（按各自峰值归一，峰值见指标表「最大像素差」列）。"""
    gray = np.abs(a_bgr.astype(np.int16) - b_bgr.astype(np.int16)).max(axis=2)
    peak = max(int(gray.max()), 1)
    norm = np.clip(gray.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _fmt_psnr(v: float) -> str:
    return "inf" if not np.isfinite(v) else f"{v:.2f}"


def compare_metrics_markdown(
    m_cross: Dict[str, float],
    m_py: Dict[str, float],
    m_onnx: Dict[str, float],
    iou: float,
    py_elapsed: float,
    onnx_elapsed: float,
    preset: str = DEFAULT_PRESET,
) -> str:
    lines = [
        "| 对比项 | MAE (/255) | 最大像素差 | 差异像素占比 | PSNR (dB) |",
        "|---|---|---|---|---|",
        f"| Python − ONNX | {m_cross['mae']:.4f} | {m_cross['max']} | {m_cross['diff_pixel_pct']:.3f}% | {_fmt_psnr(m_cross['psnr'])} |",
        f"| Python − 原图 | {m_py['mae']:.4f} | {m_py['max']} | {m_py['diff_pixel_pct']:.3f}% | {_fmt_psnr(m_py['psnr'])} |",
        f"| ONNX − 原图 | {m_onnx['mae']:.4f} | {m_onnx['max']} | {m_onnx['diff_pixel_pct']:.3f}% | {_fmt_psnr(m_onnx['psnr'])} |",
        "",
        f"高光硬掩码 IoU（Python vs ONNX）：**{iou:.4f}** ｜ "
        f"耗时：Python **{py_elapsed:.2f}s** · ONNX **{onnx_elapsed:.2f}s**",
        "",
        f"两引擎均使用「{preset}」预设的同一套参数（Python 读 {PRESETS.get(preset, PRESETS[DEFAULT_PRESET])[1].name}，"
        "ONNX 侧同一配置已内嵌模型）。"
        f"当前 ONNX 内核：{'精确内核（与 Python 逐位一致，Python − ONNX 应为 0）' if onnx_session.selected_kernel() == onnx_session.KERNEL_EXACT else 'C++ 快速内核（存在亚像素级浮点尾差）'}。"
        "热力图按各自峰值归一，峰值见「最大像素差」列。",
    ]
    return "\n".join(lines)


def single_metrics_markdown(engine: str, preset: str, out: Dict[str, Any], m_orig: Dict[str, float]) -> str:
    mask_pct = float((out["hard_mask"] > 0).mean() * 100.0)
    lines = [
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 引擎 | {engine} |",
        f"| 强度预设 | {preset} |",
        f"| 耗时 | {out['elapsed']:.2f} s |",
        f"| 高光硬掩码面积占比 | {mask_pct:.2f}% |",
        f"| 修改像素占比（vs 原图） | {m_orig['diff_pixel_pct']:.3f}% |",
        f"| 与原图 MAE (/255) | {m_orig['mae']:.4f} |",
        f"| 与原图最大像素差 | {m_orig['max']} |",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- 输入图片

def _sample_sort_key(p: Path):
    if p.stem.isdigit():
        return (0, int(p.stem), "")
    return (1, 0, p.name.lower())


def list_samples() -> List[str]:
    if not DATA_DIR.is_dir():
        return []
    files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return [p.name for p in sorted(files, key=_sample_sort_key)]


def load_sample(name: Optional[str]):
    if not name:
        return gr.skip()
    img = cv2.imread(str(DATA_DIR / name), cv2.IMREAD_COLOR)
    if img is None:
        return gr.skip()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- 主处理

def run_studio(engine: str, preset: str, image_rgb: Optional[np.ndarray]):
    """输出顺序：状态、单引擎 4 图、对比 6 图、指标面板。未涉及的组件用 gr.skip() 保持不动。"""
    skip_single = [gr.skip()] * 4
    skip_cmp = [gr.skip()] * 6
    if image_rgb is None:
        return ["请先上传图片，或从左侧下拉框选择 data/ 示例图。", *skip_single, *skip_cmp, gr.skip()]

    image_bgr = _to_bgr(image_rgb)
    orig_rgb = _to_rgb(image_bgr)
    try:
        if engine == ENGINE_COMPARE:
            py = run_python_engine(image_bgr, preset)
            onnx = run_onnx_engine(image_bgr, preset)
            m_cross = diff_metrics(py["result"], onnx["result"])
            m_py = diff_metrics(image_bgr, py["result"])
            m_onnx = diff_metrics(image_bgr, onnx["result"])
            iou = mask_iou(py["hard_mask"], onnx["hard_mask"])
            metrics_md = compare_metrics_markdown(
                m_cross, m_py, m_onnx, iou, py["elapsed"], onnx["elapsed"], preset
            )
            status = (
                f"对比完成（{preset}）：Python {py['elapsed']:.2f}s，ONNX {onnx['elapsed']:.2f}s，"
                f"MAE(Python vs ONNX) {m_cross['mae']:.4f}/255。"
            )
            if py["warnings"]:
                status += "\nPython 警告：" + "；".join(py["warnings"])
            return [
                status,
                *skip_single,
                orig_rgb,
                _to_rgb(py["result"]),
                _to_rgb(onnx["result"]),
                diff_heatmap_rgb(image_bgr, py["result"]),
                diff_heatmap_rgb(image_bgr, onnx["result"]),
                diff_heatmap_rgb(py["result"], onnx["result"]),
                metrics_md,
            ]

        run_engine = run_python_engine if engine == ENGINE_PYTHON else run_onnx_engine
        out = run_engine(image_bgr, preset)
        m_orig = diff_metrics(image_bgr, out["result"])
        status = f"{engine} 引擎（{preset}）处理完成，耗时 {out['elapsed']:.2f}s。"
        if out["warnings"]:
            status += "\n警告：" + "；".join(out["warnings"])
        return [
            status,
            orig_rgb,
            _to_rgb(out["result"]),
            out["hard_mask"],
            diff_heatmap_rgb(image_bgr, out["result"]),
            *skip_cmp,
            single_metrics_markdown(engine, preset, out, m_orig),
        ]
    except onnx_session.OnnxEngineUnavailable as exc:
        return [f"ONNX 引擎不可用：\n{exc}", *skip_single, *skip_cmp, gr.skip()]
    except Exception as exc:  # 页面不崩溃，错误进状态区
        return [f"处理失败：{exc}", *skip_single, *skip_cmp, gr.skip()]


def on_engine_change(engine: str):
    is_cmp = engine == ENGINE_COMPARE
    return gr.update(visible=not is_cmp), gr.update(visible=is_cmp)


# --------------------------------------------------------------------------- 页面

CSS = """
.gradio-container { max-width: 1500px !important; margin: 0 auto; }
#studio-header { margin-bottom: 2px; }
#studio-header p { font-size: 0.85rem; }
#status-box textarea { font-family: var(--font-mono); font-size: 0.8rem; }
#hint-note { font-size: 0.78rem; opacity: 0.75; }
"""


def build_app() -> gr.Blocks:
    samples = list_samples()
    default_sample = samples[0] if samples else None
    default_image = None
    if default_sample:
        img = cv2.imread(str(DATA_DIR / default_sample), cv2.IMREAD_COLOR)
        if img is not None:
            default_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    onnx_ok, onnx_note = onnx_session.availability()
    default_engine = ENGINE_COMPARE if onnx_ok else ENGINE_PYTHON

    with gr.Blocks(title="证件照去高光 · Python / ONNX 工作室") as demo:
        gr.Markdown(
            "## 证件照去高光 · Python / ONNX 工作室\n"
            "选择去油光强度预设后，两个引擎使用该预设的同一套参数，可切换或并排对比。",
            elem_id="studio-header",
        )
        engine = gr.Radio(
            choices=ENGINES,
            value=default_engine,
            label="推理引擎",
            info=(
                "Python = MediaPipe + OpenCV 原始链路；ONNX = 单文件模型 + 自定义算子库"
                "（默认精确内核，与 Python 逐位一致；HIGH_ONNX_KERNEL=cpp 切换 C++ 快速内核）；"
                "对比 = 两者同时运行并量化差异"
            ),
        )
        preset = gr.Radio(
            choices=list(PRESETS),
            value=DEFAULT_PRESET,
            label="去油光强度预设",
            info=(
                "强力 = high_removal_strong.onnx（去油光最强）；"
                "日常 = high_removal_daily.onnx（默认，等于当前 default.yaml 平衡效果）；"
                "保护细节 = high_removal_detail.onnx（保留更多皮肤纹理，改动最小）。"
                "Python 引擎自动加载对应 yaml，对比模式两侧参数一致。实测见 tools/THREE_ONNX_AUDIT.md"
            ),
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                input_image = gr.Image(
                    value=default_image,
                    type="numpy",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    label="输入图片",
                )
                sample_dd = gr.Dropdown(
                    choices=samples,
                    value=default_sample,
                    label="示例图片（data/）",
                )
                run_btn = gr.Button("开始处理", variant="primary")
                status = gr.Textbox(
                    value=onnx_note if not onnx_ok else "就绪。选择引擎后点击「开始处理」。",
                    label="状态",
                    lines=5,
                    interactive=False,
                    elem_id="status-box",
                )
                gr.Markdown(
                    "ONNX 引擎需要已编译的自定义算子库（见 `cpp/README.md`）：默认精确内核 "
                    "`libhigh_removal_pyops.so`（与 Python 引擎逐位一致），"
                    "`HIGH_ONNX_KERNEL=cpp` 切换 C++ 快速内核 `libhigh_removal_ops.so`。\n\n"
                    "高级参数调试请运行 `python app.py`（端口 7860）。",
                    elem_id="hint-note",
                )
            with gr.Column(scale=4):
                with gr.Group(visible=default_engine != ENGINE_COMPARE) as single_group:
                    with gr.Row():
                        single_orig = gr.Image(label="原图", interactive=False, height=360)
                        single_result = gr.Image(label="处理结果", interactive=False, height=360)
                        single_mask = gr.Image(label="高光硬掩码", interactive=False, height=360)
                        single_diff = gr.Image(label="与原图差异（热力图）", interactive=False, height=360)
                with gr.Group(visible=default_engine == ENGINE_COMPARE) as compare_group:
                    with gr.Row():
                        cmp_orig = gr.Image(label="原图", interactive=False, height=320)
                        cmp_py = gr.Image(label="Python 结果", interactive=False, height=320)
                        cmp_onnx = gr.Image(label="ONNX 结果", interactive=False, height=320)
                    with gr.Row():
                        hm_py = gr.Image(label="|Python − 原图| 热力图", interactive=False, height=320)
                        hm_onnx = gr.Image(label="|ONNX − 原图| 热力图", interactive=False, height=320)
                        hm_cross = gr.Image(label="|Python − ONNX| 热力图", interactive=False, height=320)
                metrics_md = gr.Markdown("")

        outputs = [
            status,
            single_orig, single_result, single_mask, single_diff,
            cmp_orig, cmp_py, cmp_onnx, hm_py, hm_onnx, hm_cross,
            metrics_md,
        ]
        engine.change(on_engine_change, inputs=[engine], outputs=[single_group, compare_group])
        sample_dd.change(load_sample, inputs=[sample_dd], outputs=[input_image])
        run_btn.click(run_studio, inputs=[engine, preset, input_image], outputs=outputs)
    return demo


def main() -> None:
    host = os.environ.get("HIGH_STUDIO_HOST", "").strip() or "127.0.0.1"
    port = int(os.environ.get("HIGH_STUDIO_PORT", "").strip() or "7861")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ensure_landmarker(studio_config())
        print("FaceLandmarker preloaded.")
    except Exception as exc:
        print(f"FaceLandmarker preload failed（Python 引擎首次运行时会再次尝试）: {exc}")
    demo = build_app()
    demo.launch(
        server_name=host,
        server_port=port,
        allowed_paths=[str(DATA_DIR.resolve()), str(OUTPUT_DIR.resolve())],
        theme=gr.themes.Base(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
