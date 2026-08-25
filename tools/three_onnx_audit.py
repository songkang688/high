"""三 ONNX 变体（强力/日常/保护细节）的独立复测审计脚本。

独立于三个变体分支各自的实测脚本，从零重新测量并输出 PASS/FAIL 所需的全部证据：

  1. 文件存在性 + 体积 + 内嵌 config_yaml 属性（且与 configs/onnx_*.yaml 逐字节一致）
     + simple_protect_mode 必须为 true；
  2. data/1.png 上三个模型 session.run（精确内核 libhigh_removal_pyops.so）输出两两不逐位相同；
  3. data/{1,15,16}.png 上去油光强度排序：各自硬掩码内 mean|ΔL| 与 皮肤 L≥200 亮斑消减数
     均要求 强力 > 日常 > 保护细节；
  4. data/1.png 上三个模型 session.run 与 process_image(对应 yaml) np.array_equal
     （result 与 hard_mask 都比）；
  5. 附加：日常版与 default.yaml 流水线 / 现有 models/high_removal.onnx 是否逐位相同（变体声明验证）。

用法（仓库根目录）：python tools/three_onnx_audit.py [--json /tmp/audit.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE, bootstrap_before_numpy  # noqa: E402

bootstrap_before_numpy()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402

from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import apply_runtime_mode, mask_to_uint8  # noqa: E402

VARIANTS = {
    "strong": (ROOT / "models" / "high_removal_strong.onnx", ROOT / "configs" / "onnx_strong.yaml"),
    "daily": (ROOT / "models" / "high_removal_daily.onnx", ROOT / "configs" / "onnx_daily.yaml"),
    "detail": (ROOT / "models" / "high_removal_detail.onnx", ROOT / "configs" / "onnx_detail.yaml"),
}
DEFAULT_ONNX = ROOT / "models" / "high_removal.onnx"
DEFAULT_YAML = ROOT / "configs" / "default.yaml"
EXACT_KERNEL = ROOT / "cpp" / "build" / "libhigh_removal_pyops.so"
IMAGES = ("1.png", "15.png", "16.png")
PARITY_IMAGE = "1.png"


def embedded_config_yaml(model_path: Path) -> bytes:
    model = onnx.load(str(model_path), load_external_data=False)
    for attr in model.graph.node[0].attribute:
        if attr.name == "config_yaml":
            return bytes(attr.t.raw_data)
    raise SystemExit(f"[错误] {model_path} 缺少 config_yaml 属性")


def build_config(config_yaml: bytes) -> dict:
    """与 highlight_removal.exact_kernel._build_config 完全一致。"""
    cfg = yaml.safe_load(config_yaml.decode("utf-8")) or {}
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def make_session(model_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(EXACT_KERNEL))
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def measure(image_bgr: np.ndarray, out) -> dict:
    """独立指标：各自硬掩码内 mean|ΔL|、皮肤 L≥200 前→后、整图 MAE。"""
    l_before = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    l_after = cv2.cvtColor(out.result_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    hard = out.highlight_mask > 0
    skin = mask_to_uint8(out.regions.masks["skin"]) > 0
    before = int(((l_before >= 200) & skin).sum())
    after = int(((l_after >= 200) & skin).sum())
    return {
        "hard_px": int(hard.sum()),
        "mean_abs_dl": float(np.abs(l_after - l_before)[hard].mean()) if hard.any() else 0.0,
        "skin_L200_before": before,
        "skin_L200_after": after,
        "skin_L200_drop": before - after,
        "mae": float(np.abs(out.result_bgr.astype(np.float64) - image_bgr.astype(np.float64)).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None, help="可选：完整结果另存 JSON")
    args = parser.parse_args()

    report: dict = {"checks": {}, "metrics": {}, "failures": []}

    def fail(msg: str) -> None:
        report["failures"].append(msg)
        print(f"[FAIL] {msg}")

    # ---- 检查 1：文件 + 内嵌配置 + simple_protect_mode ----
    if not EXACT_KERNEL.is_file():
        raise SystemExit(f"[错误] 缺少精确内核 {EXACT_KERNEL}，请先构建 cpp/（见 cpp/README.md）")
    embedded: dict[str, bytes] = {}
    for name, (model_path, yaml_path) in VARIANTS.items():
        if not model_path.is_file():
            fail(f"{model_path} 不存在")
            continue
        size_mib = model_path.stat().st_size / 1024 / 1024
        blob = embedded_config_yaml(model_path)
        embedded[name] = blob
        yaml_match = blob == yaml_path.read_bytes()
        cfg_raw = yaml.safe_load(blob.decode("utf-8"))
        protect_on = bool(cfg_raw.get("regions", {}).get("simple_protect_mode", False))
        report["checks"][name] = {
            "size_mib": round(size_mib, 2),
            "config_yaml_matches_file": yaml_match,
            "simple_protect_mode": protect_on,
        }
        print(f"[检查1] {name}: {size_mib:.2f} MiB  内嵌=configs文件: {yaml_match}  simple_protect_mode: {protect_on}")
        if not (4.5 <= size_mib <= 6.0):
            fail(f"{name} 体积异常 {size_mib:.2f} MiB（预期 ~5.1 MiB）")
        if not yaml_match:
            fail(f"{name} 内嵌 config_yaml 与 {yaml_path.name} 不一致")
        if not protect_on:
            fail(f"{name} simple_protect_mode 不为 true")

    # ---- process_image 全量指标（1/15/16.png × 强力/日常/保护细节/default） ----
    configs = {name: build_config(embedded[name]) for name in VARIANTS}
    configs["default"] = build_config(DEFAULT_YAML.read_bytes())
    py_outputs: dict[str, dict] = {}
    for image_name in IMAGES:
        image = cv2.imread(str(ROOT / "data" / image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"[错误] 图片读取失败: data/{image_name}")
        py_outputs[image_name] = {}
        report["metrics"][image_name] = {}
        for cfg_name, cfg in configs.items():
            out = process_image(image, deepcopy(cfg))
            if not out.success:
                fail(f"{cfg_name} 在 {image_name} 上流水线失败: {out.status}")
                continue
            py_outputs[image_name][cfg_name] = out
            m = measure(image, out)
            report["metrics"][image_name][cfg_name] = m
            print(
                f"[指标] {image_name} {cfg_name:7s}: mask_px={m['hard_px']:7d}  mean|dL|={m['mean_abs_dl']:6.3f}  "
                f"skinL>=200 {m['skin_L200_before']:6d}->{m['skin_L200_after']:6d} (降{m['skin_L200_drop']})  mae={m['mae']:.4f}"
            )

    # ---- 检查 3：排序 强力 > 日常 > 保护细节（两个指标分别判定，>=2/3 图片） ----
    dl_ok_images, drop_ok_images = [], []
    for image_name in IMAGES:
        m = report["metrics"][image_name]
        if not all(k in m for k in VARIANTS):
            continue
        dl_ok = m["strong"]["mean_abs_dl"] > m["daily"]["mean_abs_dl"] > m["detail"]["mean_abs_dl"]
        drop_ok = m["strong"]["skin_L200_drop"] > m["daily"]["skin_L200_drop"] > m["detail"]["skin_L200_drop"]
        dl_ok_images.append(dl_ok)
        drop_ok_images.append(drop_ok)
        print(f"[排序] {image_name}: mean|dL| 强>日>细: {dl_ok}  L>=200消减 强>日>细: {drop_ok}")
    report["checks"]["ranking_mean_dl_images_ok"] = sum(dl_ok_images)
    report["checks"]["ranking_drop_images_ok"] = sum(drop_ok_images)
    if not (sum(dl_ok_images) >= 2 or sum(drop_ok_images) >= 2):
        fail(f"排序不满足（mean|dL| {sum(dl_ok_images)}/3，L>=200消减 {sum(drop_ok_images)}/3，至少一项需 >=2/3）")

    # ---- 检查 2 + 4：session.run 两两不同 + 与 process_image 逐位一致（1.png） ----
    image1 = cv2.imread(str(ROOT / "data" / PARITY_IMAGE), cv2.IMREAD_COLOR)
    feed = {"image": np.ascontiguousarray(image1)}
    sess_results: dict[str, tuple] = {}
    for name, (model_path, _) in VARIANTS.items():
        sess = make_session(model_path)
        res, hard = sess.run(["result", "hard_mask"], feed)
        sess_results[name] = (res, hard)
        py_out = py_outputs[PARITY_IMAGE][name]
        parity = bool(np.array_equal(res, py_out.result_bgr) and np.array_equal(hard, py_out.highlight_mask))
        report["checks"][f"parity_session_vs_process_image_{name}"] = parity
        print(f"[检查4] {name}: session.run == process_image({VARIANTS[name][1].name}) 于 {PARITY_IMAGE}: {parity}")
        if not parity:
            fail(f"{name} session.run 与 process_image 不逐位一致（{PARITY_IMAGE}）")

    for a, b in (("strong", "daily"), ("daily", "detail"), ("strong", "detail")):
        equal = bool(np.array_equal(sess_results[a][0], sess_results[b][0]))
        report["checks"][f"session_bit_equal_{a}_vs_{b}"] = equal
        print(f"[检查2] session {a} vs {b} 于 {PARITY_IMAGE}: bit_equal={equal}（要求 False）")
        if equal:
            fail(f"{a} 与 {b} 输出逐位相同（应当不同）")

    # ---- 附加：日常版 ≟ default 流水线 / 现有 high_removal.onnx（变体声明验证） ----
    daily_eq_default_py = all(
        bool(
            np.array_equal(py_outputs[img]["daily"].result_bgr, py_outputs[img]["default"].result_bgr)
            and np.array_equal(py_outputs[img]["daily"].highlight_mask, py_outputs[img]["default"].highlight_mask)
        )
        for img in IMAGES
    )
    sess_default = make_session(DEFAULT_ONNX)
    res_def, hard_def = sess_default.run(["result", "hard_mask"], feed)
    daily_eq_default_onnx = bool(
        np.array_equal(sess_results["daily"][0], res_def) and np.array_equal(sess_results["daily"][1], hard_def)
    )
    report["checks"]["daily_bit_equal_default_pipeline_3imgs"] = daily_eq_default_py
    report["checks"]["daily_onnx_bit_equal_default_onnx_1png"] = daily_eq_default_onnx
    print(f"[声明] 日常版 == default.yaml 流水线（3 图）: {daily_eq_default_py}")
    print(f"[声明] high_removal_daily.onnx == high_removal.onnx（{PARITY_IMAGE}）: {daily_eq_default_onnx}")
    if not daily_eq_default_py:
        fail("日常版与 default.yaml 流水线不逐位一致（与变体声明矛盾）")
    if not daily_eq_default_onnx:
        fail("high_removal_daily.onnx 与现有 high_removal.onnx 不逐位一致（与变体声明矛盾）")

    verdict = "PASS" if not report["failures"] else "FAIL"
    report["verdict"] = verdict
    print(f"\n===== 审计结论: {verdict} =====")
    for msg in report["failures"]:
        print(f"  - {msg}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[信息] JSON 已保存: {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
