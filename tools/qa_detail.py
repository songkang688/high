"""保护细节（onnx_detail / high_removal_detail.onnx）独立 QA 复测脚本。

与 tools/three_onnx_audit.py 相互独立地从零验证保护细节档的全部验收项
（QA 分支 cursor/qa-onnx-detail-6544；结论见 tools/qa_detail.md）：

  1. high_removal_detail.onnx 内嵌 config_yaml 与 configs/onnx_detail.yaml 逐字节一致；
  2. data/1.png 上 session.run（精确内核）与 process_image(onnx_detail.yaml)
     np.array_equal（result 与 hard_mask 都比），且连续两次 session.run 输出一致（确定性）；
  3. data/{1,15,16}.png 上三项强度指标 detail 均须严格小于 daily：
     各自硬掩码内 mean|ΔL|、皮肤 L≥200 亮斑消减数、整图 MAE vs 原图；
  4. 内嵌配置 simple_protect_mode == true，且 protect_expand_radius ≥ daily 的取值；
  5. data/1.png 上 detail 的 session.run 输出与 daily、strong 均不逐位相同。

用法（仓库根目录）：python tools/qa_detail.py [--json /tmp/qa_detail.json]
退出码 0 = 全部 PASS。
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

MODELS = {
    "detail": ROOT / "models" / "high_removal_detail.onnx",
    "daily": ROOT / "models" / "high_removal_daily.onnx",
    "strong": ROOT / "models" / "high_removal_strong.onnx",
}
DETAIL_YAML = ROOT / "configs" / "onnx_detail.yaml"
EXACT_KERNEL = ROOT / "cpp" / "build" / "libhigh_removal_pyops.so"
IMAGES = ("1.png", "15.png", "16.png")
PARITY_IMAGE = "1.png"


def embedded_config(model_path: Path) -> bytes:
    model = onnx.load(str(model_path), load_external_data=False)
    for attr in model.graph.node[0].attribute:
        if attr.name == "config_yaml":
            return bytes(attr.t.raw_data)
    raise SystemExit(f"[错误] {model_path} 缺少 config_yaml 属性")


def runnable_config(config_yaml: bytes) -> dict:
    """与 highlight_removal.exact_kernel._build_config 相同的组装方式。"""
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
    l_before = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    l_after = cv2.cvtColor(out.result_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    hard = out.highlight_mask > 0
    skin = mask_to_uint8(out.regions.masks["skin"]) > 0
    before = int(((l_before >= 200) & skin).sum())
    after = int(((l_after >= 200) & skin).sum())
    return {
        "hard_px": int(hard.sum()),
        "mean_abs_dl": float(np.abs(l_after - l_before)[hard].mean()) if hard.any() else 0.0,
        "skin_L200_drop": before - after,
        "mae": float(np.abs(out.result_bgr.astype(np.float64) - image_bgr.astype(np.float64)).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not EXACT_KERNEL.is_file():
        raise SystemExit(f"[错误] 缺少精确内核 {EXACT_KERNEL}，请先按 cpp/README.md 构建")

    report: dict = {"checks": {}, "metrics": {}, "failures": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"][name] = bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            report["failures"].append(name)

    # ---- 1) 内嵌配置逐字节一致 ----
    blob = embedded_config(MODELS["detail"])
    check("1_embedded_config_equals_onnx_detail_yaml", blob == DETAIL_YAML.read_bytes())

    # ---- 4) 从内嵌配置读保护参数（不信任仓库 yaml 文件，直接查模型内实际生效值） ----
    cfg_detail_raw = yaml.safe_load(blob.decode("utf-8"))
    cfg_daily_raw = yaml.safe_load(embedded_config(MODELS["daily"]).decode("utf-8"))
    spm = cfg_detail_raw.get("regions", {}).get("simple_protect_mode", None)
    r_detail = cfg_detail_raw.get("regions", {}).get("protect_expand_radius", None)
    r_daily = cfg_daily_raw.get("regions", {}).get("protect_expand_radius", None)
    check("4_simple_protect_mode_true", spm is True, f"simple_protect_mode={spm}")
    check(
        "4_protect_expand_radius_not_below_daily",
        r_detail is not None and r_daily is not None and r_detail >= r_daily,
        f"detail={r_detail} daily={r_daily}",
    )

    # ---- process_image 参考输出与三项指标（detail vs daily，各自内嵌配置） ----
    cfgs = {name: runnable_config(embedded_config(path)) for name, path in MODELS.items() if name != "strong"}
    py_detail_1 = None
    for image_name in IMAGES:
        image = cv2.imread(str(ROOT / "data" / image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"[错误] 图片读取失败: data/{image_name}")
        report["metrics"][image_name] = {}
        for name, cfg in cfgs.items():
            out = process_image(image, deepcopy(cfg))
            if not out.success:
                check(f"3_pipeline_success_{name}_{image_name}", False, out.status)
                continue
            if name == "detail" and image_name == PARITY_IMAGE:
                py_detail_1 = out
            m = measure(image, out)
            report["metrics"][image_name][name] = m
            print(
                f"[指标] {image_name} {name:6s}: mask_px={m['hard_px']:7d}  mean|dL|={m['mean_abs_dl']:6.3f}  "
                f"L>=200降{m['skin_L200_drop']:6d}  mae={m['mae']:.4f}"
            )
        m = report["metrics"][image_name]
        if "detail" in m and "daily" in m:
            check(
                f"3_detail_weaker_than_daily_{image_name}",
                m["detail"]["mean_abs_dl"] < m["daily"]["mean_abs_dl"]
                and m["detail"]["skin_L200_drop"] < m["daily"]["skin_L200_drop"]
                and m["detail"]["mae"] < m["daily"]["mae"],
                (
                    f"mean|dL| {m['detail']['mean_abs_dl']:.3f}<{m['daily']['mean_abs_dl']:.3f}  "
                    f"L200降 {m['detail']['skin_L200_drop']}<{m['daily']['skin_L200_drop']}  "
                    f"mae {m['detail']['mae']:.4f}<{m['daily']['mae']:.4f}"
                ),
            )

    # ---- 2) session.run（精确内核）与 process_image 逐位一致 + 确定性 ----
    image1 = cv2.imread(str(ROOT / "data" / PARITY_IMAGE), cv2.IMREAD_COLOR)
    feed = {"image": np.ascontiguousarray(image1)}
    sess_detail = make_session(MODELS["detail"])
    res_a, hard_a = sess_detail.run(["result", "hard_mask"], feed)
    res_b, hard_b = sess_detail.run(["result", "hard_mask"], feed)
    check(
        "2_session_equals_process_image_1png",
        py_detail_1 is not None
        and np.array_equal(res_a, py_detail_1.result_bgr)
        and np.array_equal(hard_a, py_detail_1.highlight_mask),
    )
    check(
        "2_session_deterministic_two_runs",
        np.array_equal(res_a, res_b) and np.array_equal(hard_a, hard_b),
    )

    # ---- 5) 与 daily / strong 的 session 输出不逐位相同 ----
    for other in ("daily", "strong"):
        res_o, hard_o = make_session(MODELS[other]).run(["result", "hard_mask"], feed)
        check(
            f"5_not_bit_equal_to_{other}_1png",
            not (np.array_equal(res_a, res_o) and np.array_equal(hard_a, hard_o)),
        )

    verdict = "PASS" if not report["failures"] else "FAIL"
    report["verdict"] = verdict
    print(f"\n===== 保护细节 QA 结论: {verdict} =====")
    for name in report["failures"]:
        print(f"  - {name}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[信息] JSON 已保存: {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
