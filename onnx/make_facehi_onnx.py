# -*- coding: utf-8 -*-
"""生成 facehi 系列单文件 ONNX：单一 ONNX 模型封装整条去高光流水线。

采用 ONNX Runtime 官方支持的「自定义算子包装外部流水线」形态
（onnxruntime.ai/docs/reference/operators/add-custom-op.html，
"Wrapping an external inference runtime in a custom operator"，
官方工具 create_custom_op_wrapper.py 同思路）：

- 图中只有一个自定义域 ai.facehi 的节点 HighlightRemoval；
- 两个子模型（face_detector.onnx / face_landmarks_detector.onnx）、
  合并后的模式配置 YAML、Haar 兜底级联 XML，全部序列化进节点属性；
- 运行需要注册配套的自定义算子库 libfacehi_custom_ops.so（ORT 设计如此，
  任何含自定义域的 ONNX 都必须带 kernel 实现库）。

预设（--preset）：
  full    默认。onnx/models/facehi.onnx：内嵌常用/高保真/最高质量三种模式
          合并配置，节点属性 mode=常用模式（强力去高光）。
  detail  git-high2/models/facehi_detail.onnx：保护细节版。内嵌单一
          「保护细节」模式（检测=弱、修复=削弱、method=混合 + 纹理优先键），
          节点属性 mode=保护细节。去高光故意弱一点，换取五官/皮肤纹理保留。

用法：
  .venv/bin/python onnx/make_facehi_onnx.py                 # 生成 facehi.onnx
  .venv/bin/python onnx/make_facehi_onnx.py --preset detail # 生成 facehi_detail.onnx
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
MODES = ("常用模式", "高保真模式", "最高质量模式")
DETAIL_MODE = "保护细节"
CONFIG_SECTIONS = ("face_detection", "regions", "highlight_detection",
                   "highlight_removal", "pipeline")

# 各预设：输出路径 + 烘焙进节点属性的 mode。
PRESETS = {
    "full": {"out": MODELS_DIR / "facehi.onnx", "mode": "常用模式"},
    "detail": {"out": ROOT / "git-high2" / "models" / "facehi_detail.onnx",
               "mode": DETAIL_MODE},
}


def build_config_yaml() -> str:
    """full 预设：用 cli_process.load_cli_config 生成三种模式合并配置（与 Python CLI 完全一致）。"""
    from cli_process import load_cli_config

    modes = {}
    for mode in MODES:
        cfg = load_cli_config(mode)
        modes[mode] = {sec: cfg.get(sec, {}) for sec in CONFIG_SECTIONS}
    doc = {"default_mode": "常用模式", "modes": modes}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def build_detail_config() -> dict:
    """detail 预设：「保护细节」模式合并配置。

    组合思路（去高光故意弱一点，换五官/皮肤纹理尽量保留）：
    - 底座与常用/高保真相同：cli_process.load_cli_config 的 pipeline 固定四键
      + process_scale=compromise（取高保真模式底座 = 检测/修复均为「正常」）；
    - 检测 ←「弱」预设（configs/detection_sensitivity_presets.yaml）：阈值抬高、
      掩码不膨胀反腐蚀、区域面积上限收紧 → 改动像素显著减少、掩码更贴核心；
    - 修复 ←「削弱」预设（configs/removal_intensity_presets.yaml）：压制强度
      0.94→0.68、最终混合 0.97→0.82、inpaint 半径 6→3 → 高光残留可略多；
    - 在削弱之上把 method 改回「混合」并叠加纹理优先键（参考
      configs/experimental_better.yaml 的思路，只用现有 C++ 内核支持的键）：
      * extreme_core_extra_l: 30 —— Telea 强修复核心阈值抬到 lab_l_threshold+30
        （弱检测下 198+30=228，只处理近饱和白斑），其余油光全走保真 Lab 压制，
        避免 Telea 大面积平滑抹掉皮肤纹理；
      * texture_preserve_strength: 1.0 —— 保真压制的高频纹理回加拉满
        （内核系数 0.16×strength 封顶）；
      * edge_protect_strength: 0.60 —— 边缘保护再抬一档，纹理边界少动；
    - regions 保护区加宽：protect_expand_radius 5→7、eye_protect_extra_radius
      1.0→1.6 → 眼鼻嘴保护区更干净。
    """
    from cli_process import (INTENSITY_PATH, SENSITIVITY_PATH, _merge_preset,
                             load_cli_config)

    cfg = load_cli_config("高保真模式")
    cfg["highlight_detection"] = _merge_preset(
        cfg.get("highlight_detection", {}), SENSITIVITY_PATH, "弱")
    cfg["highlight_removal"] = _merge_preset(
        cfg.get("highlight_removal", {}), INTENSITY_PATH, "削弱")
    cfg["highlight_removal"].update({
        "mode": "混合",
        "extreme_core_extra_l": 30,
        "texture_preserve_strength": 1.0,
        "edge_protect_strength": 0.60,
    })
    cfg["regions"].update({
        "protect_expand_radius": 7,
        "eye_protect_extra_radius": 1.6,
    })
    return cfg


def build_detail_config_yaml() -> str:
    cfg = build_detail_config()
    doc = {"default_mode": DETAIL_MODE,
           "modes": {DETAIL_MODE: {sec: cfg.get(sec, {}) for sec in CONFIG_SECTIONS}}}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def bytes_tensor(name: str, data: bytes) -> onnx.TensorProto:
    arr = np.frombuffer(data, dtype=np.uint8)
    return helper.make_tensor(name, TensorProto.UINT8, [len(arr)], arr.tobytes(), raw=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="full",
                    help="full=facehi.onnx（常用模式）；detail=facehi_detail.onnx（保护细节）")
    ap.add_argument("--out", default=None, help="覆盖默认输出路径（可选）")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    out_path = Path(args.out) if args.out else preset["out"]
    mode = preset["mode"]

    detector = (MODELS_DIR / "face_detector.onnx").read_bytes()
    landmarks = (MODELS_DIR / "face_landmarks_detector.onnx").read_bytes()

    import cv2

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    haar_xml = haar_path.read_bytes() if haar_path.is_file() else b""

    if args.preset == "detail":
        config_yaml = build_detail_config_yaml()
        node_doc = ("整条面部去高光流水线（保护细节版）：BlazeFace 检测→478点关键点→分区→"
                    "弱敏感度高光检测→削弱混合修复（Telea 仅近饱和核心）→加宽保护回写。"
                    "纹理优先：去高光故意弱一点，换五官/皮肤纹理尽量保留。")
        graph_doc = (
            "facehi 保护细节版单一 ONNX 入口（内嵌「保护细节」模式：检测=弱、修复=削弱、"
            "method=混合、Telea 核心阈值 +30、纹理回加拉满、保护区加宽）。"
            "输入 image：uint8 BGR [H,W,3]（与 cv2.imread 一致）；"
            "输出 result：uint8 BGR 同形状，highlight_mask：uint8 [H,W]。"
            "必须注册 libfacehi_custom_ops 自定义算子库后才能创建会话。"
        )
        model_doc = ("面部去高光整条流水线的单一 ONNX 封装——保护细节版"
                     "（ORT 自定义算子包装外部流水线；mode=保护细节）")
    else:
        config_yaml = build_config_yaml()
        node_doc = "整条面部去高光流水线：BlazeFace 检测→478点关键点→分区→高光检测→修复→保护回写"
        graph_doc = (
            "facehi 单一 ONNX 入口。输入 image：uint8 BGR [H,W,3]（与 cv2.imread 一致）；"
            "输出 result：uint8 BGR 同形状，highlight_mask：uint8 [H,W]。"
            "必须注册 libfacehi_custom_ops 自定义算子库后才能创建会话。"
        )
        model_doc = "面部去高光整条流水线的单一 ONNX 封装（ORT 自定义算子包装外部流水线）"

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
        mode=mode,
        doc_string=node_doc,
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
        name="facehi" if args.preset == "full" else "facehi_detail",
        inputs=[image_vi, lm_vi],
        outputs=[result_vi, mask_vi],
        initializer=[lm_default],
        doc_string=graph_doc,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("ai.facehi", 1)],
        producer_name="facehi",
        doc_string=model_doc,
    )
    model.ir_version = 9
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[完成] {out_path}（{size_mb:.2f} MB，preset={args.preset}，mode={mode}，"
          f"内嵌 detector={len(detector)}B, landmarks={len(landmarks)}B, "
          f"haar={len(haar_xml)}B）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
