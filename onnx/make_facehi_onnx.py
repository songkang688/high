# -*- coding: utf-8 -*-
"""生成 onnx/models/facehi.onnx：单一 ONNX 模型封装整条去高光流水线。

采用 ONNX Runtime 官方支持的「自定义算子包装外部流水线」形态
（onnxruntime.ai/docs/reference/operators/add-custom-op.html，
"Wrapping an external inference runtime in a custom operator"，
官方工具 create_custom_op_wrapper.py 同思路）：

- 图中只有一个自定义域 ai.facehi 的节点 HighlightRemoval；
- 两个子模型（face_detector.onnx / face_landmarks_detector.onnx）、
  按 cli_process.load_cli_config 合并后的三种模式配置 YAML、
  Haar 兜底级联 XML，全部序列化进节点属性；
- 运行需要注册配套的自定义算子库 libfacehi_custom_ops.so（ORT 设计如此，
  任何含自定义域的 ONNX 都必须带 kernel 实现库）。

用法：
  .venv/bin/python onnx/make_facehi_onnx.py
  .venv/bin/python onnx/make_facehi_onnx.py --out git-high2/output-1.22.0/facehi.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "onnx" / "models"
DEFAULT_OUT_PATH = MODELS_DIR / "facehi.onnx"
MODES = ("常用模式", "高保真模式", "最高质量模式")
CONFIG_SECTIONS = ("face_detection", "regions", "highlight_detection",
                   "highlight_removal", "pipeline")


def build_config_yaml() -> str:
    """用 cli_process.load_cli_config 生成各模式合并后的配置（与 Python CLI 完全一致）。"""
    from cli_process import load_cli_config

    modes = {}
    for mode in MODES:
        cfg = load_cli_config(mode)
        modes[mode] = {sec: cfg.get(sec, {}) for sec in CONFIG_SECTIONS}
    doc = {"default_mode": "常用模式", "modes": modes}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def bytes_tensor(name: str, data: bytes) -> onnx.TensorProto:
    arr = np.frombuffer(data, dtype=np.uint8)
    return helper.make_tensor(name, TensorProto.UINT8, [len(arr)], arr.tobytes(), raw=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 facehi.onnx（不覆盖则请传 --out）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                    help="输出路径。默认 onnx/models/facehi.onnx；ORT 1.22 交付请指到 output-1.22.0/")
    args = ap.parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detector = (MODELS_DIR / "face_detector.onnx").read_bytes()
    landmarks = (MODELS_DIR / "face_landmarks_detector.onnx").read_bytes()

    import cv2

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    haar_xml = haar_path.read_bytes() if haar_path.is_file() else b""

    config_yaml = build_config_yaml()

    node = helper.make_node(
        "HighlightRemoval",
        inputs=["image", "landmarks"],
        outputs=["result", "highlight_mask"],
        name="facehi_highlight_removal",
        domain="ai.facehi",
        detector_onnx=bytes_tensor("detector_onnx", detector),
        landmarks_onnx=bytes_tensor("landmarks_onnx", landmarks),
        haar_xml=bytes_tensor("haar_xml", haar_xml),
        config_yaml=config_yaml,
        mode="常用模式",
        doc_string="整条面部去高光流水线：BlazeFace 检测→478点关键点→分区→高光检测→修复→保护回写",
    )

    # image: uint8 BGR [H,W,3]（也接受 [1,H,W,3]，故不固定秩）。
    image_vi = helper.make_value_info(
        "image", helper.make_tensor_type_proto(TensorProto.UINT8, None))
    result_vi = helper.make_value_info(
        "result", helper.make_tensor_type_proto(TensorProto.UINT8, None))
    mask_vi = helper.make_value_info(
        "highlight_mask", helper.make_tensor_type_proto(TensorProto.UINT8, None))
    # landmarks 是带默认 initializer（空 [0,3]）的可选输入：不喂则走内置人脸检测。
    lm_vi = helper.make_value_info(
        "landmarks", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, 3]))
    lm_default = helper.make_tensor("landmarks", TensorProto.FLOAT, [0, 3], b"", raw=True)

    graph = helper.make_graph(
        nodes=[node],
        name="facehi",
        inputs=[image_vi, lm_vi],
        outputs=[result_vi, mask_vi],
        initializer=[lm_default],
        doc_string=(
            "facehi 单一 ONNX 入口。输入 image：uint8 BGR [H,W,3]（与 cv2.imread 一致）；"
            "输出 result：uint8 BGR 同形状，highlight_mask：uint8 [H,W]。"
            "必须注册 libfacehi_custom_ops 自定义算子库后才能创建会话。"
        ),
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("ai.facehi", 1)],
        producer_name="facehi",
        producer_version=f"onnx-{onnx.__version__}",
        doc_string="面部去高光整条流水线的单一 ONNX 封装（ORT 自定义算子包装外部流水线）",
    )
    model.ir_version = 9
    meta = model.metadata_props.add()
    meta.key = "onnx_package"
    meta.value = onnx.__version__
    meta2 = model.metadata_props.add()
    meta2.key = "target_onnxruntime"
    meta2.value = "1.22.0"
    onnx.save(model, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[完成] {out_path}（{size_mb:.2f} MB，onnx={onnx.__version__}，"
          f"内嵌 detector={len(detector)}B, landmarks={len(landmarks)}B, haar={len(haar_xml)}B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
