"""对比 models/high_removal.onnx（default）与 models/high_removal_strong.onnx（强力）的去油光强度。

配置直接从两个 ONNX 文件内嵌的 config_yaml 属性字节提取，走
highlight_removal.exact_kernel 的同一条构建/执行路径（与 session.run +
libhigh_removal_pyops 逐位相同，见 exact_kernel.py 模块注释），因此数字
就是这两个 ONNX 模型的真实输出。

指标（每张图、每个模型）：
  1. mask_mean_abs_dL —— 各自硬掩码内 |ΔL|（LAB L 通道，0-255）均值；
  2. skin_L200_before / after —— 皮肤掩码内 L>=200 像素个数（处理前→后）；
  3. mae —— 全图三通道对原图的平均绝对误差。

用法：python3 tools/compute_strong_metrics.py [--images data/1.png data/15.png data/16.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import DEFAULT_RUNTIME_MODE  # noqa: E402
from highlight_removal.utils import apply_runtime_mode  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402


def embedded_config_yaml(model_path: Path) -> bytes:
    model = onnx.load(str(model_path), load_external_data=False)
    node = model.graph.node[0]
    for attr in node.attribute:
        if attr.name == "config_yaml":
            return bytes(attr.t.raw_data)
    raise SystemExit(f"[错误] {model_path} 中没有 config_yaml 属性")


def build_config(config_yaml: bytes) -> dict:
    """与 highlight_removal.exact_kernel._build_config 逐行一致。"""
    import yaml

    cfg = yaml.safe_load(config_yaml.decode("utf-8")) or {}
    cfg["runtime"] = apply_runtime_mode(DEFAULT_RUNTIME_MODE)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    return cfg


def lab_l(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)


def run_one(image_bgr: np.ndarray, cfg: dict) -> dict:
    out = process_image(image_bgr, deepcopy(cfg))
    if not out.success:
        raise SystemExit(f"[错误] 流水线失败: {out.status}")
    result = np.ascontiguousarray(out.result_bgr, dtype=np.uint8)
    hard_mask = np.ascontiguousarray(out.highlight_mask, dtype=np.uint8)
    skin = (out.regions["skin"] > 0) if out.regions is not None else np.zeros(image_bgr.shape[:2], bool)

    l_before = lab_l(image_bgr)
    l_after = lab_l(result)
    mask = hard_mask > 0
    d_l = np.abs(l_after - l_before)

    return {
        "mask_pixels": int(mask.sum()),
        "mask_mean_abs_dL": float(d_l[mask].mean()) if mask.any() else 0.0,
        "skin_L200_before": int(((l_before >= 200) & skin).sum()),
        "skin_L200_after": int(((l_after >= 200) & skin).sum()),
        "mae": float(np.abs(result.astype(np.float64) - image_bgr.astype(np.float64)).mean()),
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", default=[str(ROOT / "data" / n) for n in ("1.png", "15.png", "16.png")])
    parser.add_argument("--default-model", default=str(ROOT / "models" / "high_removal.onnx"))
    parser.add_argument("--strong-model", default=str(ROOT / "models" / "high_removal_strong.onnx"))
    parser.add_argument("--json", default=None, help="可选：结果另存为 JSON")
    args = parser.parse_args()

    cfg_default = build_config(embedded_config_yaml(Path(args.default_model)))
    cfg_strong = build_config(embedded_config_yaml(Path(args.strong_model)))

    report: dict = {}
    for path_str in args.images:
        path = Path(path_str)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"[错误] 图片读取失败: {path}")
        m_default = run_one(image, cfg_default)
        m_strong = run_one(image, cfg_strong)
        bit_equal = bool(np.array_equal(m_default.pop("result"), m_strong.pop("result")))
        report[path.name] = {"default": m_default, "strong": m_strong, "outputs_bit_equal": bit_equal}

        print(f"\n=== {path.name} ({image.shape[1]}x{image.shape[0]}) ===")
        for label, m in (("default", m_default), ("strong ", m_strong)):
            print(
                f"  {label}: mask_px={m['mask_pixels']:7d}  mean|dL|={m['mask_mean_abs_dL']:6.3f}  "
                f"skin L>=200: {m['skin_L200_before']:6d} -> {m['skin_L200_after']:6d}  mae={m['mae']:.4f}"
            )
        print(f"  outputs_bit_equal={bit_equal}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[信息] JSON 已保存: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
