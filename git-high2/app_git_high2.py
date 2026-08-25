# -*- coding: utf-8 -*-
"""git-high2：Python / ONNX 去高光对比工作台（端口 7862，避开 app.py 7860 / app_compare.py 7861）。

三种工作模式（顶部标签切换）：
  - Python 版：highlight_removal.pipeline.process_image + cli_process.load_cli_config
    （常用模式 / 高保真模式 / 最高质量模式；需要在仓库内运行）
  - ONNX 版：git-high2/models/facehi*.onnx + libfacehi_custom_ops 单会话推理
    （会话按档位懒加载并缓存；档位烘焙在各 onnx 文件内：facehi.onnx=默认常用模式，
    facehi_strong.onnx=强力版·去高光最狠）
  - 对比：同一份解码数组同时跑两条链路，四宫格（原图 / Python / ONNX / 放大差分）
    并给出 MAE / 最大像素差 / 差异像素占比 / PSNR / 硬掩码 IoU / 两边耗时

启动：python git-high2/app_git_high2.py  →  http://127.0.0.1:7862

本文件放在自包含目录 git-high2/ 内：整夹拷到本机（如 ~/git-high2）后
ONNX 版与对比页的 ONNX 链路仍可独立运行；Python 版需要完整仓库
（highlight_removal/、cli_process.py、configs/、models/face_landmarker.task）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_repo_root() -> Path | None:
    """向上查找完整仓库根（含 highlight_removal 包与 cli_process.py）。"""
    for parent in (_HERE, *_HERE.parents):
        if (parent / "highlight_removal" / "pipeline.py").is_file() and \
           (parent / "cli_process.py").is_file():
            return parent
    return None


REPO_ROOT = _find_repo_root()
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 与 app.py / app_compare.py 一致：线程与设备环境必须在 numpy 之前设好
if REPO_ROOT is not None:
    from highlight_removal.runtime_bootstrap import bootstrap_before_numpy

    bootstrap_before_numpy()

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # ONNX 版不可用时 Python 版仍可正常工作
    ort = None

from facehi_onnx import MODEL_VARIANTS, FacehiOnnx, find_model, find_ops_lib

# Python 原始流水线：仅在完整仓库内可用；独立目录运行时优雅降级
PY_PIPELINE_ERROR: Optional[str] = None
if REPO_ROOT is not None:
    try:
        from cli_process import MODE_CHOICES, load_cli_config, preload_models
        from highlight_removal.pipeline import process_image
    except Exception as exc:  # noqa: BLE001
        PY_PIPELINE_ERROR = f"Python 流水线导入失败：{exc}"
else:
    PY_PIPELINE_ERROR = (
        "当前为 git-high2 独立目录运行（未找到仓库根的 highlight_removal 包），"
        "Python 版不可用；ONNX 版不受影响。要使用 Python 版请在完整仓库内启动。"
    )
if PY_PIPELINE_ERROR is not None:
    MODE_CHOICES = ("常用模式", "高保真模式", "最高质量模式")

DATA_DIR = (REPO_ROOT / "data") if REPO_ROOT is not None else (_HERE / "data")
OUTPUT_DIR = _HERE / "runtime_outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODE = "常用模式"
DIFF_BASE_CHOICES = ["Python vs ONNX", "原图 vs Python", "原图 vs ONNX"]
DEFAULT_DIFF_GAIN = 4  # 与 highlight_removal.pipeline 差分图风格一致：clip(absdiff * 4)

# ONNX 档位 = 独立 onnx 文件（档位烘焙在图节点属性里，加载后不可切换）。
# 标签 → facehi_onnx.MODEL_VARIANTS 里的档位别名；四档全列，
# models/ 下缺对应文件时运行报友好提示（不崩），补上文件即可用。
_ONNX_VARIANT_LABELS = [
    ("facehi.onnx（默认 · 常用模式）", "default"),
    ("facehi_strong.onnx（强力 · 去高光最狠）", "strong"),
    ("facehi_daily.onnx（日常）", "daily"),
    ("facehi_detail.onnx（保护细节 · 纹理/五官优先）", "detail"),
]
ONNX_VARIANT_BY_LABEL = dict(_ONNX_VARIANT_LABELS)
ONNX_VARIANT_CHOICES = list(ONNX_VARIANT_BY_LABEL)
DEFAULT_ONNX_VARIANT_LABEL = ONNX_VARIANT_CHOICES[0]

ONNX_MODE_NOTE = (
    "处理模式仅对 Python 版生效；ONNX 输出档位由所选 onnx 文件烘焙决定"
    "（见下方「ONNX 模型档位」）。"
)

BUILD_HELP_MD = """**未找到自定义算子库 `libfacehi_custom_ops.so`。** ONNX 版需要先编译一次（详见 `git-high2/README.md` 与 `cpp/README.md`）：

```bash
# 1. 安装构建依赖（Ubuntu）
sudo apt install cmake g++ libyaml-cpp-dev

# 2. 下载 ONNX Runtime CPU 预编译包（只需要头文件）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz
tar xzf onnxruntime-linux-x64-1.29.0.tgz -C /opt && sudo mv /opt/onnxruntime-linux-x64-1.29.0 /opt/ort

# 3. 编译 OpenCV 4.14 静态库（数值对齐要求，见 README）后构建本库
cmake -B git-high2/cpp/build -S git-high2/cpp -DCMAKE_BUILD_TYPE=Release \\
      -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 -DORT_ROOT=/opt/ort
cmake --build git-high2/cpp/build -j
cp git-high2/cpp/build/libfacehi_custom_ops.so git-high2/lib/
```

编译完成后无需重启本页面，直接重新点击运行即可。"""


# ---------------------------------------------------------------------------
# 中文路径安全的读写（np.fromfile + imdecode / imencode + tofile）
# ---------------------------------------------------------------------------

def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_image_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image_bgr: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise IOError(f"结果编码失败: {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# ONNX 会话：全局懒加载并缓存（只 register 一次，不随图片重建）
# ---------------------------------------------------------------------------

class OnnxUnavailableError(RuntimeError):
    """ONNX 链路不可用（缺 onnxruntime / 缺 .so / 缺模型），message 为中文说明。"""


_ONNX_LOCK = threading.Lock()
_ONNX_CACHE: Dict[str, Dict[str, Any]] = {}  # variant -> {"wrapper", "load_seconds"}


def get_onnx_wrapper(variant: str = "default") -> FacehiOnnx:
    if ort is None:
        raise OnnxUnavailableError("未安装 onnxruntime，请先执行：`pip install onnxruntime`")
    with _ONNX_LOCK:
        cached = _ONNX_CACHE.get(variant)
        if cached is not None:
            return cached["wrapper"]
        if find_model(variant) is None:
            name = MODEL_VARIANTS.get(variant, variant)
            env_key = ("FACEHI_ONNX_MODEL" if variant == "default"
                       else f"FACEHI_ONNX_MODEL_{variant.upper()}")
            raise OnnxUnavailableError(
                f"未找到模型 `git-high2/models/{name}`（档位 {variant}）。"
                "该档位的 onnx 可能尚未生成/拷贝：请确认 models/ 子目录包含该文件，"
                f"或设置环境变量 {env_key} 指向模型文件；其它档位不受影响。"
            )
        if find_ops_lib() is None:
            raise OnnxUnavailableError(BUILD_HELP_MD)
        t0 = time.perf_counter()
        wrapper = FacehiOnnx(variant=variant)
        _ONNX_CACHE[variant] = {"wrapper": wrapper,
                                "load_seconds": time.perf_counter() - t0}
        return wrapper


# ---------------------------------------------------------------------------
# 推理与指标
# ---------------------------------------------------------------------------

def _decode_input(image_path: Optional[str]) -> np.ndarray:
    """统一入口解码：np.fromfile + cv2.imdecode，中文路径 / 中文文件名安全。"""
    if not image_path:
        raise gr.Error("请先上传图片或选择样例图片。")
    bgr = _read_image_bgr(Path(image_path))
    if bgr is None:
        raise gr.Error(f"图片读取失败：{Path(image_path).name}（支持 jpg / png / bmp / webp）")
    return np.ascontiguousarray(bgr, dtype=np.uint8)


def _run_python(bgr: np.ndarray, mode_name: str):
    if PY_PIPELINE_ERROR is not None:
        raise RuntimeError(PY_PIPELINE_ERROR)
    cfg = load_cli_config(mode_name if mode_name in MODE_CHOICES else DEFAULT_MODE)
    t0 = time.perf_counter()
    out = process_image(bgr, cfg)
    return out, time.perf_counter() - t0


def _run_onnx(bgr: np.ndarray, variant: str = "default") -> Tuple[np.ndarray, np.ndarray, float]:
    wrapper = get_onnx_wrapper(variant)
    t0 = time.perf_counter()
    result, mask = wrapper.run(bgr)
    return result, mask, time.perf_counter() - t0


def _pixel_metrics(py_img: np.ndarray, onnx_img: np.ndarray) -> Dict[str, float]:
    """与 onnx/tests/compare_facehi_onnx.py 相同口径。"""
    d = np.abs(py_img.astype(np.int16) - onnx_img.astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    return {
        "mae": float(d.mean()),
        "max": int(d.max()),
        "diff_pixel_pct": float((d.max(axis=2) > 0).mean() * 100.0),
        "psnr": float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
    }


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a1 = np.asarray(a).squeeze() > 0
    b1 = np.asarray(b).squeeze() > 0
    union = int(np.logical_or(a1, b1).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a1, b1).sum() / union)


def _amplified_diff(a: np.ndarray, b: np.ndarray, gain: float) -> np.ndarray:
    diff = cv2.absdiff(a, b).astype(np.float32)
    return np.clip(diff * float(gain), 0, 255).astype(np.uint8)


def _diff_view(a: np.ndarray, b: np.ndarray, gain: float, auto: bool) -> Tuple[np.ndarray, str]:
    """生成差分可视化与倍数说明。auto=True 时按最大像素差归一，保证差异位置可见。"""
    max_d = int(cv2.absdiff(a, b).max())
    if auto:
        if max_d == 0:
            return np.zeros_like(a), "两图逐位一致，无差异像素。"
        gain_eff = min(255.0 / max_d, 128.0)
        note = f"差分放大倍数：自动 ×{gain_eff:.0f}（最大像素差 {max_d}/255）"
    else:
        gain_eff = float(gain)
        note = f"差分放大倍数：×{gain_eff:.0f}（最大像素差 {max_d}/255，流水线差分图默认 ×4）"
    return _amplified_diff(a, b, gain_eff), note


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return _bgr_to_rgb(mask)


def _save_result(prefix: str, image_bgr: np.ndarray) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUTPUT_DIR / f"{prefix}_{stamp}.png"
    _write_image(path, image_bgr)  # imencode + tofile，中文路径安全
    try:
        return os.path.relpath(path, _HERE).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 页面回调
# ---------------------------------------------------------------------------

def _list_samples() -> list[str]:
    if not DATA_DIR.is_dir():
        return []

    def _sort_key(p: Path):
        if p.stem.isdigit():
            return (0, int(p.stem), p.suffix.lower())
        return (1, p.name.lower())

    files = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return [f"data/{p.name}" for p in sorted(files, key=_sort_key)]


def select_sample(rel_path: Optional[str]):
    if not rel_path:
        return gr.update()
    path = (DATA_DIR / Path(rel_path).name).resolve()
    if not path.is_file():
        raise gr.Error(f"样例不存在：{rel_path}")
    return str(path)


def ui_run_python(image_path, mode_name):
    bgr = _decode_input(image_path)
    try:
        out, elapsed = _run_python(bgr, mode_name)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"**Python 版不可用**：{exc}"
    h, w = bgr.shape[:2]
    if not out.success:
        return _bgr_to_rgb(bgr), None, f"**处理失败**：{out.status}"
    saved = _save_result("python_result", out.result_bgr)
    report = (
        f"**耗时 {elapsed:.3f} 秒** ｜ 模式：{mode_name} ｜ 输出 {w}×{h} ｜ "
        f"检测到人脸 {out.detection_count} 个 ｜ 已保存 `{saved}`"
    )
    return _bgr_to_rgb(out.result_bgr), _mask_rgb(out.highlight_mask), report


def ui_run_onnx(image_path, _mode_name, onnx_variant_label):
    variant = ONNX_VARIANT_BY_LABEL.get(onnx_variant_label, "default")
    bgr = _decode_input(image_path)
    try:
        result, mask, elapsed = _run_onnx(bgr, variant)
    except OnnxUnavailableError as exc:
        return None, None, str(exc)
    except Exception as exc:  # noqa: BLE001 —— 会话内部错误按原因展示，不静默崩溃
        return None, None, f"**ONNX 会话运行失败**：{exc}"
    h, w = bgr.shape[:2]
    saved = _save_result(f"onnx_result_{variant}", result)
    load_note = ""
    cached = _ONNX_CACHE.get(variant)
    if cached is not None and cached["load_seconds"] is not None:
        load_note = f" ｜ 会话加载 {cached['load_seconds']:.2f} 秒（仅首次）"
    report = (
        f"**耗时 {elapsed:.3f} 秒** ｜ 模型：{MODEL_VARIANTS.get(variant, variant)}"
        f"（档位烘焙在图内） ｜ 输出 {w}×{h} ｜ 已保存 `{saved}`{load_note}"
    )
    return _bgr_to_rgb(result), _mask_rgb(mask), report


def _psnr_text(value: float) -> str:
    return "∞（逐位一致）" if not np.isfinite(value) else f"{value:.2f} dB"


def _metrics_table_html(m: Dict[str, float], iou: float, t_py: float, t_onnx: float) -> str:
    rows = [
        ("MAE（平均绝对误差 /255）", f"{m['mae']:.4f}"),
        ("最大像素差（/255）", f"{m['max']}"),
        ("差异像素占比", f"{m['diff_pixel_pct']:.3f}%"),
        ("PSNR", _psnr_text(m["psnr"])),
        ("硬掩码 IoU", f"{iou:.4f}"),
        ("Python 耗时", f"{t_py:.3f} 秒"),
        ("ONNX 耗时", f"{t_onnx:.3f} 秒"),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td class='num'>{value}</td></tr>" for name, value in rows
    )
    return (
        "<div class='hl-metrics'>"
        "<table><thead><tr><th>指标（Python vs ONNX）</th><th>数值</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        "<p class='hl-caption'>口径与 onnx/tests/compare_facehi_onnx.py 一致；"
        "两次处理使用同一份解码数组，不存在重复解码差异。</p>"
        "</div>"
    )


def _pick_diff_pair(state: Dict[str, Any], diff_base: str):
    if not state:
        return None, None
    mapping = {
        "Python vs ONNX": ("py", "onnx"),
        "原图 vs Python": ("orig", "py"),
        "原图 vs ONNX": ("orig", "onnx"),
    }
    key_a, key_b = mapping.get(diff_base, ("py", "onnx"))
    return state.get(key_a), state.get(key_b)


def ui_update_diff(state, diff_base, diff_gain, diff_auto):
    a, b = _pick_diff_pair(state or {}, diff_base)
    if a is None or b is None:
        return gr.update(), gr.update()
    diff_img, note = _diff_view(a, b, diff_gain, diff_auto)
    return _bgr_to_rgb(diff_img), note


def ui_run_compare(image_path, mode_name, onnx_variant_label, diff_base, diff_gain, diff_auto):
    variant = ONNX_VARIANT_BY_LABEL.get(onnx_variant_label, "default")
    bgr = _decode_input(image_path)
    orig_rgb = _bgr_to_rgb(bgr)
    state: Dict[str, Any] = {"orig": bgr, "py": None, "onnx": None}
    notes: list[str] = []

    py_rgb = None
    py_mask = None
    t_py = None
    try:
        out, t_py = _run_python(bgr, mode_name)
        if out.success:
            state["py"] = out.result_bgr
            py_mask = out.highlight_mask
            py_rgb = _bgr_to_rgb(out.result_bgr)
            notes.append(f"Python：耗时 {t_py:.3f} 秒（{mode_name}）")
        else:
            notes.append(f"Python 处理失败：{out.status}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Python 不可用：{exc}")

    onnx_mask = None
    t_onnx = None
    onnx_rgb = None
    onnx_error_md = None
    try:
        onnx_result, onnx_mask, t_onnx = _run_onnx(bgr, variant)
        state["onnx"] = onnx_result
        onnx_rgb = _bgr_to_rgb(onnx_result)
        notes.append("ONNX：耗时 {:.3f} 秒（{}）".format(
            t_onnx, MODEL_VARIANTS.get(variant, variant)))
    except OnnxUnavailableError as exc:
        onnx_error_md = str(exc)
        notes.append("ONNX：不可用（见下方说明）")
    except Exception as exc:  # noqa: BLE001
        onnx_error_md = f"**ONNX 会话运行失败**：{exc}"
        notes.append("ONNX：运行失败（见下方说明）")

    if state["py"] is not None and state["onnx"] is not None:
        metrics_html = _metrics_table_html(
            _pixel_metrics(state["py"], state["onnx"]),
            _mask_iou(py_mask, onnx_mask),
            t_py,
            t_onnx,
        )
    else:
        metrics_html = (
            "<div class='hl-metrics'><p class='hl-caption'>"
            "两条链路未同时成功，暂无对比指标。</p></div>"
        )

    diff_a, diff_b = _pick_diff_pair(state, diff_base)
    diff_rgb = None
    diff_note = ""
    if diff_a is not None and diff_b is not None:
        diff_img, diff_note = _diff_view(diff_a, diff_b, diff_gain, diff_auto)
        diff_rgb = _bgr_to_rgb(diff_img)

    report = " ｜ ".join(notes)
    if onnx_error_md:
        report += "\n\n" + onnx_error_md
    return orig_rgb, py_rgb, onnx_rgb, diff_rgb, diff_note, metrics_html, report, state


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

APP_CSS = """
:root {
    --hl-accent: #1d4ed8;
    --hl-accent-hover: #1e40af;
    --hl-bg: #f7f7f8;
    --hl-panel: #ffffff;
    --hl-border: #e4e4e7;
    --hl-text: #18181b;
    --hl-dim: #71717a;
}
.gradio-container {
    background: var(--hl-bg) !important;
    max-width: 1560px !important;
    margin: 0 auto !important;
    font-family: system-ui, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif !important;
}
.gradio-container * { box-shadow: none !important; }
footer { display: none !important; }

#hl-header { border-bottom: 1px solid var(--hl-border); padding-bottom: 4px; margin-bottom: 6px; gap: 0 !important; }
#hl-header h1 { font-size: 20px; font-weight: 600; color: var(--hl-text); margin: 6px 0 0; }
#hl-header p { font-size: 13px; color: var(--hl-dim); margin: 4px 0 8px; }

/* 顶部标签：唯一强调色用在选中态 */
.tab-nav { border-bottom: 1px solid var(--hl-border) !important; }
.tab-nav button {
    font-size: 15px !important; color: var(--hl-dim) !important;
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--hl-accent) !important; font-weight: 600 !important;
    border-bottom: 2px solid var(--hl-accent) !important;
}
.tabitem { border: none !important; background: transparent !important; padding: 10px 0 0 !important; }

/* 面板：白底 + 细边框，无阴影无装饰 */
.block, .form, .gradio-accordion {
    background: var(--hl-panel) !important;
    border: 1px solid var(--hl-border) !important;
    border-radius: 6px !important;
}
.gap.panel, .styler { background: transparent !important; border: none !important; }

button.primary {
    background: var(--hl-accent) !important; color: #ffffff !important;
    border: 1px solid var(--hl-accent) !important; border-radius: 6px !important;
    font-weight: 500 !important;
}
button.primary:hover { background: var(--hl-accent-hover) !important; }
button.secondary { border-radius: 6px !important; }

label span, .block label { font-size: 13px !important; color: var(--hl-dim) !important; }

/* 紧凑的辅助说明 */
.hl-note { font-size: 12.5px; color: var(--hl-dim); line-height: 1.7; }
.hl-note p { margin: 4px 0; }
.hl-note code, .hl-report code {
    background: #f4f4f5; border: 1px solid var(--hl-border);
    padding: 0 4px; border-radius: 3px; font-size: 12px;
}
.hl-report { font-size: 13.5px; line-height: 1.8; color: var(--hl-text); }
.hl-report p { margin: 6px 0; }

/* 指标表：数值右对齐等宽字体 */
.hl-metrics table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.hl-metrics th, .hl-metrics td {
    padding: 7px 12px; border-bottom: 1px solid var(--hl-border); text-align: left;
}
.hl-metrics th { color: var(--hl-dim); font-weight: 500; font-size: 12.5px; }
.hl-metrics td.num {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Courier New", monospace;
    text-align: right; color: var(--hl-text);
}
.hl-caption { font-size: 12px; color: var(--hl-dim); margin: 8px 2px 2px; }
"""


def _env_status_md() -> str:
    ops_path = find_ops_lib()
    if ops_path is not None:
        try:
            ops_text = f"已就绪（`{os.path.relpath(ops_path, _HERE)}`）"
        except ValueError:
            ops_text = f"已就绪（`{ops_path}`）"
    else:
        ops_text = "未编译，ONNX 版页面内含构建说明"
    model_lines = []
    for variant, name in MODEL_VARIANTS.items():
        path = find_model(variant)
        state = f"已就绪（{path.stat().st_size / 1e6:.1f} MB）" if path is not None else "缺失"
        model_lines.append(f"`models/{name}`：{state}")
    ort_text = ort.__version__ if ort is not None else "未安装（`pip install onnxruntime`）"
    py_text = "已就绪（完整仓库）" if PY_PIPELINE_ERROR is None else "不可用（独立目录运行）"
    return (
        "**运行环境**\n\n"
        f"onnxruntime：{ort_text}\n\n"
        f"OpenCV：{cv2.__version__} ｜ Gradio：{gr.__version__}\n\n"
        + "\n\n".join(model_lines) + "\n\n"
        f"自定义算子库：{ops_text}\n\n"
        f"Python 原始流水线：{py_text}"
    )


_GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def _make_theme():
    from gradio.themes.utils import fonts as _fonts

    font_names = ["system-ui", "PingFang SC", "Microsoft YaHei", "sans-serif"]
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.zinc,
        font=[_fonts.Font(n) for n in font_names],
    ).set(
        button_primary_background_fill="#1d4ed8",
        button_primary_background_fill_hover="#1e40af",
        button_primary_text_color="#ffffff",
        block_shadow="none",
    )


def _style_kwargs() -> dict:
    return {"css": APP_CSS, "theme": _make_theme()}


def build_app() -> gr.Blocks:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _list_samples()
    default_sample = "data/1.png" if "data/1.png" in samples else (samples[0] if samples else None)
    default_image = (
        str((DATA_DIR / Path(default_sample).name).resolve()) if default_sample else None
    )

    # Gradio 6 起 css/theme 移到 launch()；旧版仍在 Blocks 构造器
    blocks_kwargs = {} if _GRADIO_MAJOR >= 6 else _style_kwargs()

    with gr.Blocks(title="git-high2 · Python / ONNX 去高光对比", **blocks_kwargs) as demo:
        cmp_state = gr.State(None)

        with gr.Column(elem_id="hl-header"):
            gr.Markdown(
                "# git-high2 · 证件照去高光 · Python / ONNX 对比工作台\n"
                "同一套去高光算法的两种交付形态：Python 原始流水线（MediaPipe + OpenCV）"
                "与单文件模型 `git-high2/models/facehi.onnx`（`ai.facehi:HighlightRemoval` "
                "自定义算子，内嵌子模型与配置）。支持 jpg / png / bmp / webp，中文路径可用。"
            )

        with gr.Row():
            # 左栏：输入与运行环境（紧凑的辅助信息）
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(
                    label="输入图片（上传或选择样例）",
                    type="filepath",
                    sources=["upload"],
                    value=default_image,
                    height=280,
                )
                sample_dd = gr.Dropdown(
                    choices=samples,
                    value=default_sample,
                    label="样例图片（data/ 目录）",
                )
                mode_radio = gr.Radio(
                    choices=list(MODE_CHOICES),
                    value=DEFAULT_MODE,
                    label="处理模式",
                    info=ONNX_MODE_NOTE,
                )
                onnx_variant_dd = gr.Dropdown(
                    choices=ONNX_VARIANT_CHOICES,
                    value=DEFAULT_ONNX_VARIANT_LABEL,
                    label="ONNX 模型档位（独立 onnx 文件，档位烘焙在图内）",
                    info="强力=去高光最狠；保护细节=改动最少、纹理/五官优先"
                         "（见 models/DETAIL.md）；缺对应文件的档位运行时会提示。",
                )
                gr.Markdown(_env_status_md(), elem_classes=["hl-note"])

            # 右栏：三种工作模式（主内容，占大空间）
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Python 版"):
                        py_btn = gr.Button("运行 Python 版", variant="primary")
                        with gr.Row():
                            py_result = gr.Image(label="去高光结果", interactive=False, height=520, scale=3)
                            py_mask = gr.Image(label="高光硬掩码", interactive=False, height=520, scale=2)
                        py_report = gr.Markdown("", elem_classes=["hl-report"])

                    with gr.Tab("ONNX 版"):
                        onnx_btn = gr.Button("运行 ONNX 版", variant="primary")
                        with gr.Row():
                            onnx_result = gr.Image(label="去高光结果", interactive=False, height=520, scale=3)
                            onnx_mask = gr.Image(label="高光硬掩码", interactive=False, height=520, scale=2)
                        onnx_report = gr.Markdown("", elem_classes=["hl-report"])

                    with gr.Tab("对比"):
                        cmp_btn = gr.Button("运行对比（Python + ONNX）", variant="primary")
                        with gr.Row():
                            cmp_orig = gr.Image(label="原图", interactive=False, height=380)
                            cmp_py = gr.Image(label="Python 结果", interactive=False, height=380)
                        with gr.Row():
                            cmp_onnx = gr.Image(label="ONNX 结果", interactive=False, height=380)
                            cmp_diff = gr.Image(label="差分可视化（|a − b| × 放大倍数）", interactive=False, height=380)
                        cmp_diff_note = gr.Markdown("", elem_classes=["hl-note"])
                        cmp_metrics = gr.HTML("")
                        cmp_report = gr.Markdown("", elem_classes=["hl-report"])
                        with gr.Accordion("高级选项：差分显示", open=False):
                            diff_base = gr.Dropdown(
                                choices=DIFF_BASE_CHOICES,
                                value=DIFF_BASE_CHOICES[0],
                                label="差分基准",
                            )
                            diff_auto = gr.Checkbox(
                                value=True,
                                label="自动放大（按最大像素差归一，保证差异位置可见）",
                            )
                            diff_gain = gr.Slider(
                                1, 64, value=DEFAULT_DIFF_GAIN, step=1,
                                label="手动放大倍数（关闭自动放大后生效；×4 与流水线差分图一致）",
                            )

        sample_dd.change(select_sample, inputs=[sample_dd], outputs=[input_image])
        input_image.upload(lambda: gr.update(value=None), None, [sample_dd])

        py_btn.click(
            ui_run_python,
            inputs=[input_image, mode_radio],
            outputs=[py_result, py_mask, py_report],
        )
        onnx_btn.click(
            ui_run_onnx,
            inputs=[input_image, mode_radio, onnx_variant_dd],
            outputs=[onnx_result, onnx_mask, onnx_report],
        )
        cmp_btn.click(
            ui_run_compare,
            inputs=[input_image, mode_radio, onnx_variant_dd, diff_base, diff_gain, diff_auto],
            outputs=[cmp_orig, cmp_py, cmp_onnx, cmp_diff, cmp_diff_note, cmp_metrics, cmp_report, cmp_state],
        )
        diff_inputs = [cmp_state, diff_base, diff_gain, diff_auto]
        diff_base.change(ui_update_diff, inputs=diff_inputs, outputs=[cmp_diff, cmp_diff_note])
        diff_auto.change(ui_update_diff, inputs=diff_inputs, outputs=[cmp_diff, cmp_diff_note])
        diff_gain.release(ui_update_diff, inputs=diff_inputs, outputs=[cmp_diff, cmp_diff_note])

    return demo


if __name__ == "__main__":
    # 与 app.py 一致：仓库内启动时预加载 FaceLandmarker，首次处理不用等模型初始化
    if PY_PIPELINE_ERROR is None:
        if preload_models(load_cli_config(DEFAULT_MODE)):
            print("FaceLandmarker preloaded.")
        else:
            print("FaceLandmarker preload failed（Python 版将在首次运行时重试）")
    else:
        print(PY_PIPELINE_ERROR)
    app = build_app()
    allowed = [str(OUTPUT_DIR.resolve()), str(_HERE.resolve())]
    if DATA_DIR.is_dir():
        allowed.append(str(DATA_DIR.resolve()))
    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=7862,
        allowed_paths=allowed,
    )
    if _GRADIO_MAJOR >= 6:
        launch_kwargs.update(_style_kwargs())
    app.launch(**launch_kwargs)
