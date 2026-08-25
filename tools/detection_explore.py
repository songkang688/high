"""最终复测（PART 2）多角度探索实验：预设强度、替代算法、速度。

实验内容（全部真实执行、输出实测数字，结果并入 docs/FINAL_DETECTION_REPORT.md）：
  A. 去高光强度预设对比（正常 / 强力 / 削弱）× 代表图 1 / 15 / 16：
     - 先验证「强力」预设与 default.yaml 的 highlight_removal 是否为同一组参数；
     - 每套预设报告：掩码面积、修改区平均 |ΔL| / ΔE / Δa / Δb、质量审计警告、
       残余高光（对输出图按同一套默认配置**重新检测**得到的高光面积）。
  C. 替代修复算法（同一检测掩码、同一保护逻辑，仅替换修复核心）× 图 1 / 15：
     - guided   ：导向滤波（cv2.ximgproc.guidedFilter）作为肤色参考，替代 TELEA+双边混合；
     - freqsep  ：频率分离（低频取 inpaint 基底、高频纹理 100% 回加），替代 texture_keep 0.16×；
     - param    ：仅调参数（brightness_suppress 0.94→0.80、chroma_restore 0.38→0.55），走原流水线。
     - 评价指标：残余高光面积（重新检测）、修改区 ΔE / |ΔL|、非高光皮肤 ΔE（附带伤害）。
  D. 速度：data/1.png 上 raw process_image / 精确内核 session.run / C++ 内核 session.run
     （各预热 1 次后取 3 次中位数），以及 C++ CLI high_onnx_session 整进程耗时。

调试图输出到 runtime_outputs/detection_explore/（已 gitignore），
文字汇总输出到 runtime_outputs/detection_explore/explore_summary.md。

用法：python tools/detection_explore.py
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
from copy import deepcopy
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
from highlight_removal.remove_highlight import _edge_protected_alpha, _inpaint_lab_reference  # noqa: E402
from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE  # noqa: E402
from highlight_removal.utils import apply_runtime_mode, load_yaml, mask_to_uint8  # noqa: E402

OUT_DIR = ROOT / "runtime_outputs" / "detection_explore"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def default_cfg() -> dict:
    cfg = load_yaml(ROOT / "configs" / "default.yaml")
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def preset_cfg(preset_name: str) -> dict:
    """default.yaml + removal_intensity_presets.yaml 中指定预设（正常 = 不改）。"""
    cfg = default_cfg()
    if preset_name != "正常":
        data = load_yaml(ROOT / "configs" / "removal_intensity_presets.yaml")
        cfg["highlight_removal"] = {
            **cfg.get("highlight_removal", {}),
            **((data.get("presets") or {}).get(preset_name) or {}),
        }
    return cfg


def lab_delta_stats(orig: np.ndarray, result: np.ndarray, region: np.ndarray) -> dict:
    """region（bool）内的 LAB 变化统计。OpenCV LAB uint8：L 0~255（=L*×2.55），a/b 偏移 128。"""
    lab0 = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab1 - lab0
    if not region.any():
        return {"mean_abs_dL": 0.0, "mean_dL": 0.0, "mean_abs_da": 0.0, "mean_abs_db": 0.0, "mean_dE": 0.0, "px": 0}
    dl, da, db = d[:, :, 0][region], d[:, :, 1][region], d[:, :, 2][region]
    de = np.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2)[region]
    return {
        "mean_abs_dL": float(np.abs(dl).mean()),
        "mean_dL": float(dl.mean()),
        "mean_abs_da": float(np.abs(da).mean()),
        "mean_abs_db": float(np.abs(db).mean()),
        "mean_dE": float(de.mean()),
        "px": int(region.sum()),
    }


def redetect(result_bgr: np.ndarray, cfg: dict) -> tuple[int, float]:
    """对输出图重新跑同一套流水线，返回（残余高光硬掩码像素数, 占人脸比例）。"""
    out = process_image(result_bgr, cfg)
    if not out.success:
        return -1, float("nan")
    area = int((out.highlight_mask > 0).sum())
    ratio = float(out.metrics.get("高光 mask 面积占人脸比例", float("nan")))
    return area, ratio


def gated_masks(out) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 _remove_highlight_core 的口径（皮肤内、非保护区）对硬/软掩码做同样裁剪。"""
    skin = mask_to_uint8(out.regions.masks["skin"]) > 0
    protect = mask_to_uint8(out.regions.masks["protect"]) > 0
    hard = mask_to_uint8(out.highlight_mask).copy()
    soft = mask_to_uint8(out.soft_mask).copy()
    hard[(~skin) | protect] = 0
    soft[(~skin) | protect] = 0
    return hard, soft, skin, protect


# --------------------------------------------------------------------- 替代修复核心

def variant_reference_removal(image_bgr: np.ndarray, hard: np.ndarray, soft: np.ndarray,
                              params: dict, ref_mode: str) -> np.ndarray:
    """与 _faithful_suppress 同构，仅替换肤色参考基底 / 纹理回加策略。

    ref_mode:
      guided  = 导向滤波参考（边缘感知平滑，替代 TELEA inpaint + 双边混合）
      freqsep = 频率分离（低频用 TELEA 基底的低频，高频纹理 100% 回加，替代 0.16× texture_keep）
    """
    brightness_strength = float(params.get("brightness_suppress_strength", 0.74))
    color_strength = float(params.get("chroma_restore_strength", 0.30))
    texture_strength = float(params.get("texture_preserve_strength", 0.78))
    edge_strength = float(params.get("edge_protect_strength", 0.45))
    final_alpha = float(params.get("final_blend_alpha", 0.92))
    radius = int(params.get("inpainting_radius", 3))
    luminance_floor = float(params.get("faithful_luminance_floor", 0.86))

    alpha0 = _edge_protected_alpha(image_bgr, soft, edge_strength)
    alpha_l = (alpha0 * brightness_strength * final_alpha).clip(0, 1)
    alpha_c = (alpha0 * color_strength * final_alpha).clip(0, 1)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    low = cv2.GaussianBlur(L, (0, 0), 2.2)
    texture = L - low

    if ref_mode == "guided":
        # 导向滤波：以原图自身为 guide 的边缘感知平滑，作为肤色基线。
        guided = cv2.ximgproc.guidedFilter(guide=image_bgr, src=image_bgr, radius=16, eps=15.0 * 15.0)
        ref_lab = cv2.cvtColor(guided, cv2.COLOR_BGR2LAB).astype(np.float32)
        texture_keep = texture * (0.16 * np.clip(texture_strength, 0, 1))  # 纹理策略与现有一致
    elif ref_mode == "freqsep":
        ref_lab = _inpaint_lab_reference(image_bgr, hard, radius)  # 基底与现有一致
        ref_lab[:, :, 0] = cv2.GaussianBlur(ref_lab[:, :, 0], (0, 0), 2.2)  # 只取低频
        texture_keep = texture  # 高频 100% 回加
    else:
        raise ValueError(ref_mode)

    ref_L = ref_lab[:, :, 0]
    local_skin = cv2.GaussianBlur(L, (0, 0), max(8.0, float(params.get("local_sigma", 6.0)) * 1.6))
    min_allowed = np.maximum(L * luminance_floor, local_skin * 0.93)
    target_L = np.maximum(ref_L + texture_keep, min_allowed)
    target_L = np.maximum(target_L, local_skin - 5.0)
    target_L = np.minimum(target_L, L + 0.5)

    out_lab = lab.copy()
    out_lab[:, :, 0] = L * (1 - alpha_l) + target_L * alpha_l
    out_lab[:, :, 1] = lab[:, :, 1] * (1 - alpha_c) + ref_lab[:, :, 1] * alpha_c
    out_lab[:, :, 2] = lab[:, :, 2] * (1 - alpha_c) + ref_lab[:, :, 2] * alpha_c
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    keep = (alpha0 <= 0.001)[..., None]
    return np.where(keep, image_bgr, out).astype(np.uint8)


def eval_result(tag: str, name: str, img: np.ndarray, result: np.ndarray,
                hard_gated: np.ndarray, skin: np.ndarray, soft_gated: np.ndarray,
                cfg_for_redetect: dict, warnings: list[str] | None = None) -> dict:
    hl = hard_gated > 0
    clean_skin = skin & (soft_gated == 0)
    s_hl = lab_delta_stats(img, result, hl)
    s_clean = lab_delta_stats(img, result, clean_skin)
    # 绝对残余油光指标（检测器是区域分位数自适应的，重检面积不反映真实残余，
    # 这里补充绝对口径：掩码内平均 L 前→后、皮肤内 L≥178（lab_l_threshold）亮斑像素数前→后）。
    L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    L1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    mean_L_before = float(L0[hl].mean()) if hl.any() else float("nan")
    mean_L_after = float(L1[hl].mean()) if hl.any() else float("nan")
    p95_before = float(np.percentile(L0[hl], 95)) if hl.any() else float("nan")
    p95_after = float(np.percentile(L1[hl], 95)) if hl.any() else float("nan")
    bright_before = int(((L0 >= 200) & skin).sum())
    bright_after = int(((L1 >= 200) & skin).sum())
    leftover_px, leftover_ratio = redetect(result, cfg_for_redetect)
    out_path = OUT_DIR / f"{name}_{tag}.png"
    cv2.imwrite(str(out_path), result)
    row = {
        "tag": tag, "name": name,
        "hl_px": int(hl.sum()),
        "hl_mean_abs_dL": s_hl["mean_abs_dL"], "hl_mean_dL": s_hl["mean_dL"],
        "hl_mean_abs_da": s_hl["mean_abs_da"], "hl_mean_abs_db": s_hl["mean_abs_db"],
        "hl_mean_dE": s_hl["mean_dE"],
        "clean_skin_dE": s_clean["mean_dE"],
        "mean_L_before": mean_L_before, "mean_L_after": mean_L_after,
        "p95_before": p95_before, "p95_after": p95_after,
        "bright200_before": bright_before, "bright200_after": bright_after,
        "leftover_px": leftover_px, "leftover_ratio": leftover_ratio,
        "warnings": warnings or [],
    }
    log(f"  [{tag}] {name}: 掩码内 |ΔL|={row['hl_mean_abs_dL']:.2f} ΔE={row['hl_mean_dE']:.2f} "
        f"|Δa|={row['hl_mean_abs_da']:.2f} |Δb|={row['hl_mean_abs_db']:.2f} | "
        f"掩码内均L {mean_L_before:.1f}→{mean_L_after:.1f}，P95 {p95_before:.0f}→{p95_after:.0f} | "
        f"皮肤L≥200 {bright_before}→{bright_after}px | "
        f"非高光皮肤 ΔE={row['clean_skin_dE']:.4f} | "
        f"重检面积占脸 {leftover_ratio*100:.2f}% | 警告 {len(row['warnings'])} 条")
    return row


def save_montage(name: str, img: np.ndarray, hard_gated: np.ndarray, variants: list[tuple[str, np.ndarray]]) -> None:
    """高光掩码 bbox 附近的横向拼图（原图 + 各变体），供人工观感检查。"""
    ys, xs = np.where(hard_gated > 0)
    if len(ys) == 0:
        return
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    tiles = [img[y0:y1, x0:x1]] + [r[y0:y1, x0:x1] for _, r in variants]
    sep = np.full((y1 - y0, 4, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, sep, t])
    labels = "orig_" + "_".join(t for t, _ in variants)
    cv2.imwrite(str(OUT_DIR / f"{name}_montage_{labels}.png"), row)


# --------------------------------------------------------------------- 实验 A：预设强度

def experiment_a() -> None:
    log("\n## 实验 A：去高光强度预设（正常 / 强力 / 削弱）\n")
    normal = preset_cfg("正常")["highlight_removal"]
    strong = preset_cfg("强力")["highlight_removal"]
    weak = preset_cfg("削弱")["highlight_removal"]
    same = {k: strong.get(k) for k in strong} == {k: normal.get(k) for k in strong if k in normal} and all(
        normal.get(k) == v for k, v in strong.items()
    )
    log(f"- 「强力」预设与 default.yaml（=「正常」）逐项相等：{same}")
    diffs = {k: (normal.get(k), weak.get(k)) for k in weak if normal.get(k) != weak.get(k)}
    log(f"- 「削弱」与「正常」差异项：{len(diffs)} 项（mode {normal.get('mode')}→{weak.get('mode')} 等）")

    redetect_cfg = default_cfg()
    for stem in ("1", "15", "16"):
        img = cv2.imread(str(ROOT / "data" / f"{stem}.png"), cv2.IMREAD_COLOR)
        log(f"\n### data/{stem}.png")
        base_out = process_image(img, default_cfg())
        hard_g, soft_g, skin, _ = gated_masks(base_out)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_L_before = float(orig_lab[:, :, 0][hard_g > 0].mean()) if (hard_g > 0).any() else float("nan")
        log(f"  掩码内原始平均 L = {mean_L_before:.1f}（/255）")
        variants = []
        for preset in ("正常", "强力", "削弱"):
            cfg = preset_cfg(preset)
            out = process_image(img, cfg)
            # 削弱预设的检测配置相同 → 掩码相同；口径统一用正常套的 gated 掩码评估
            eval_result(f"presetA_{preset}", stem, img, out.result_bgr, hard_g, skin, soft_g,
                        redetect_cfg, out.warnings)
            variants.append((preset, out.result_bgr))
        save_montage(stem, img, hard_g, [v for v in variants if v[0] != "强力"])
        eq = np.array_equal(variants[0][1], variants[1][1])
        log(f"  「正常」与「强力」输出逐位相同：{eq}")


# --------------------------------------------------------------------- 实验 C：替代算法

def experiment_c() -> None:
    log("\n## 实验 C：替代修复算法（同一检测掩码与保护逻辑，仅换修复核心）\n")
    redetect_cfg = default_cfg()
    for stem in ("1", "15"):
        img = cv2.imread(str(ROOT / "data" / f"{stem}.png"), cv2.IMREAD_COLOR)
        log(f"\n### data/{stem}.png")
        cfg = default_cfg()
        base_out = process_image(img, cfg)
        hard_g, soft_g, skin, protect = gated_masks(base_out)
        params = {**cfg.get("highlight_removal", {}), **cfg.get("highlight_detection", {})}

        # 基线 = 现有 混合 模式完整流水线输出
        eval_result("baseline", stem, img, base_out.result_bgr, hard_g, skin, soft_g,
                    redetect_cfg, base_out.warnings)
        variants = [("baseline", base_out.result_bgr)]

        # guided / freqsep：替换修复核心（不含 混合 模式的极端核心二次 inpaint，属保真同构）
        for mode in ("guided", "freqsep"):
            res = variant_reference_removal(img, hard_g, soft_g, params, mode)
            immutable = ((soft_g == 0) | protect | (~skin))[..., None]
            res = np.where(immutable, img, res).astype(np.uint8)
            eval_result(mode, stem, img, res, hard_g, skin, soft_g, redetect_cfg)
            variants.append((mode, res))

        # param：仅调参数走原流水线（brightness_suppress 0.94→0.80、chroma_restore 0.38→0.55）
        cfg_p = default_cfg()
        cfg_p["highlight_removal"]["brightness_suppress_strength"] = 0.80
        cfg_p["highlight_removal"]["chroma_restore_strength"] = 0.55
        out_p = process_image(img, cfg_p)
        eval_result("param", stem, img, out_p.result_bgr, hard_g, skin, soft_g,
                    redetect_cfg, out_p.warnings)
        variants.append(("param", out_p.result_bgr))
        save_montage(stem, img, hard_g, variants)

        # 残余高光掩码调试图（基线）
        left_out = process_image(base_out.result_bgr, redetect_cfg)
        cv2.imwrite(str(OUT_DIR / f"{stem}_baseline_leftover_mask.png"), left_out.highlight_mask)


# --------------------------------------------------------------------- 实验 D：速度

def _median_time(fn, n: int = 3) -> float:
    fn()  # 预热
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def experiment_d() -> None:
    log("\n## 实验 D：速度（data/1.png，预热 1 次后 3 次取中位数）\n")
    img = cv2.imread(str(ROOT / "data" / "1.png"), cv2.IMREAD_COLOR)
    cfg = default_cfg()
    model = str(ROOT / "models" / "high_removal.onnx")

    t_py = _median_time(lambda: process_image(img, cfg))
    log(f"- raw process_image（Python 流水线本体）：{t_py*1000:.0f} ms")

    for kernel, lib in (("精确内核", "libhigh_removal_pyops.so"), ("C++ 内核", "libhigh_removal_ops.so")):
        so = ort.SessionOptions()
        so.register_custom_ops_library(str(ROOT / "cpp" / "build" / lib))
        sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        t = _median_time(lambda: sess.run(["result", "hard_mask"], {"image": np.ascontiguousarray(img)}))
        log(f"- 单 ONNX session.run（{kernel} {lib}）：{t*1000:.0f} ms")

    cli = ROOT / "cpp" / "build" / "high_onnx_session"
    if cli.is_file():
        def run_cli():
            subprocess.run(
                [str(cli), "--model", model, "--ops", str(ROOT / "cpp" / "build" / "libhigh_removal_ops.so"),
                 "--input", str(ROOT / "data" / "1.png"), "--output", str(OUT_DIR / "speed_cli_out.png")],
                check=True, capture_output=True,
                env={"LD_LIBRARY_PATH": "/opt/ort/onnxruntime-linux-x64-1.22.0/lib", "PATH": "/usr/bin:/bin"},
            )
        t_cli = _median_time(run_cli)
        log(f"- C++ CLI high_onnx_session 整进程（含加载模型/库）：{t_cli*1000:.0f} ms")


def main() -> int:
    cfg = default_cfg()
    if not preload_models(cfg):
        return 3
    log("# 多角度探索实验实测记录（tools/detection_explore.py 自动生成）")
    experiment_a()
    experiment_c()
    experiment_d()
    (OUT_DIR / "explore_summary.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n汇总已写入 {OUT_DIR / 'explore_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
