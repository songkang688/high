# -*- coding: utf-8 -*-
"""生成 facehi*.onnx：单一 ONNX 模型封装整条去高光流水线。

采用 ONNX Runtime 官方支持的「自定义算子包装外部流水线」形态
（onnxruntime.ai/docs/reference/operators/add-custom-op.html，
"Wrapping an external inference runtime in a custom operator"，
官方工具 create_custom_op_wrapper.py 同思路）：

- 图中只有一个自定义域 ai.facehi 的节点 HighlightRemoval；
- 两个子模型（face_detector.onnx / face_landmarks_detector.onnx）、
  按 cli_process.load_cli_config 合并后的模式配置 YAML、
  Haar 兜底级联 XML，全部序列化进节点属性；
- 节点属性 mode 决定运行时实际生效的档位（烘焙进图，加载后不可切换）；
- 运行需要注册配套的自定义算子库 libfacehi_custom_ops.so（ORT 设计如此，
  任何含自定义域的 ONNX 都必须带 kernel 实现库）。

用法：
  # 默认：生成 onnx/models/facehi.onnx（烘焙 mode=常用模式，与历史行为一致）
  python onnx/make_facehi_onnx.py

  # 烘焙任一标准模式（常用模式 / 高保真模式 / 最高质量模式）
  python onnx/make_facehi_onnx.py --mode 常用模式 --out git-high2/models/facehi.onnx

  # 强力版（三独立档位之一）：检测=灵敏 + 修复=强力 + method=混合 +
  # process_scale=compromise（即现有常用模式），并在此之上加强修复/混合参数
  python onnx/make_facehi_onnx.py --preset strong --out git-high2/models/facehi_strong.onnx
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
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
CONFIG_SECTIONS = ("face_detection", "regions", "highlight_detection",
                   "highlight_removal", "pipeline")

# ---------------------------------------------------------------------------
# 独立档位预设（每个预设 = 基准标准模式 + 少量差异项，烘焙为一个新 mode）。
# 本文件当前实现强力版；日常版 / 保护细节版由各自分支补充各自的 preset。
# ---------------------------------------------------------------------------

# 强力版：基准就是现有「常用模式」（检测=灵敏、修复=强力、method=混合、
# process_scale=compromise，当前默认里去高光最狠的档位），在其上加强
# faithful_suppress / strong_inpaint 的混合参数（同一套
# ai.facehi C++ kernel，不换算法内核）：
#   - 关键杠杆 extreme_core_extra_l 10→0：混合模式的强修复（Telea inpaint）
#     分支从「极亮核心」扩大到硬掩码内全部 L≥lab_l_threshold 的像素，
#     高光被周围皮肤修复填充而不只是压亮度；
#   - 亮度压制 / Poisson 混合 / 最终混合 alpha 拉满或接近拉满；
#   - 纹理保留与边缘保护降低（细节保护最少）；
#   - inpainting 半径 6→8（kernel 内 clamp 上限 9）；
#   - 质量守卫上限相应放宽（守卫只发警告，不回退结果）。
PRESETS: dict[str, dict] = {
    "strong": {
        "mode_name": "强力模式",
        "base_mode": "常用模式",
        "default_out": "facehi_strong.onnx",
        "overrides": {
            "highlight_removal": {
                "brightness_suppress_strength": 1.0,
                "chroma_restore_strength": 0.46,
                "texture_preserve_strength": 0.55,
                "edge_protect_strength": 0.28,
                "inpainting_radius": 8,
                "poisson_alpha_strength": 0.95,
                "final_blend_alpha": 1.0,
                "faithful_luminance_floor": 0.62,
                "extreme_core_extra_l": 0,
                "max_allowed_modify_area_ratio": 0.28,
                "max_allowed_mean_brightness_change": 30,
                "max_allowed_local_color_delta": 20,
            },
        },
        "doc": "强力版：去高光最狠、细节保护最少（常用模式基础上加强修复与混合，强修复分支覆盖整个高光核心）",
    },
}


def build_modes_config() -> dict:
    """用 cli_process.load_cli_config 生成各标准模式合并后的配置（与 Python CLI 完全一致）。"""
    from cli_process import load_cli_config

    modes = {}
    for mode in MODES:
        cfg = load_cli_config(mode)
        modes[mode] = {sec: cfg.get(sec, {}) for sec in CONFIG_SECTIONS}
    return modes


def apply_preset(modes: dict, preset: dict) -> str:
    """基于基准模式生成预设档位配置，追加进 modes；返回新 mode 名。"""
    mode_name = preset["mode_name"]
    cfg = deepcopy(modes[preset["base_mode"]])
    for section, values in preset["overrides"].items():
        cfg.setdefault(section, {}).update(values)
    modes[mode_name] = cfg
    return mode_name


def config_yaml_text(modes: dict, default_mode: str) -> str:
    doc = {"default_mode": default_mode, "modes": modes}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def bytes_tensor(name: str, data: bytes) -> onnx.TensorProto:
    arr = np.frombuffer(data, dtype=np.uint8)
    return helper.make_tensor(name, TensorProto.UINT8, [len(arr)], arr.tobytes(), raw=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 facehi 单文件 ONNX（可选烘焙档位）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mode", choices=MODES, default=None,
                       help="烘焙的标准模式（默认：常用模式）")
    group.add_argument("--preset", choices=sorted(PRESETS), default=None,
                       help="烘焙的独立档位预设（如 strong=强力版），与 --mode 互斥")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出路径（默认 onnx/models/facehi.onnx；"
                             "--preset 时默认 onnx/models/<preset 对应文件名>）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    detector = (MODELS_DIR / "face_detector.onnx").read_bytes()
    landmarks = (MODELS_DIR / "face_landmarks_detector.onnx").read_bytes()

    import cv2

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    haar_xml = haar_path.read_bytes() if haar_path.is_file() else b""

    modes = build_modes_config()
    if args.preset:
        preset = PRESETS[args.preset]
        baked_mode = apply_preset(modes, preset)
        out_path = args.out or (MODELS_DIR / preset["default_out"])
        doc_extra = preset["doc"]
    else:
        baked_mode = args.mode or "常用模式"
        out_path = args.out or (MODELS_DIR / "facehi.onnx")
        doc_extra = f"烘焙档位：{baked_mode}"
    config_yaml = config_yaml_text(modes, default_mode=baked_mode)

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
        mode=baked_mode,
        doc_string=("整条面部去高光流水线：BlazeFace 检测→478点关键点→分区→高光检测→修复→保护回写；"
                    + doc_extra),
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
            f"烘焙档位 mode={baked_mode}。"
        ),
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("ai.facehi", 1)],
        producer_name="facehi",
        doc_string=("面部去高光整条流水线的单一 ONNX 封装（ORT 自定义算子包装外部流水线）；"
                    + doc_extra),
    )
    model.ir_version = 9
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[完成] {out_path}（{size_mb:.2f} MB，mode={baked_mode}，"
          f"内嵌 detector={len(detector)}B, landmarks={len(landmarks)}B, "
          f"haar={len(haar_xml)}B, modes={list(modes)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
