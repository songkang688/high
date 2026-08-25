# -*- coding: utf-8 -*-
"""生成 git-high2/models/facehi*.onnx：单一 ONNX 模型封装整条去高光流水线。

与仓库根 onnx/make_facehi_onnx.py 同源，多了 --preset：同一套 C++
`ai.facehi:HighlightRemoval` 自定义算子内核（lib/libfacehi_custom_ops.so），
只是把不同的 mode / 配置烘焙进节点属性，产出互相独立的 onnx 文件：

  --preset standard  facehi.onnx        烘焙「常用模式」（强力向，检测=灵敏、修复=强力）
  --preset daily     facehi_daily.onnx  烘焙独立的「日常模式」（日常/平衡）：
                                        高光检测与修复的每个差异参数都取
                                        强力档（facehi_strong.onnx，mode=强力模式）与
                                        保护细节档（facehi_detail.onnx，mode=保护细节）
                                        两端实际烘焙值的中值；method=混合、
                                        process_scale=compromise。

采用 ONNX Runtime 官方支持的「自定义算子包装外部流水线」形态
（onnxruntime.ai/docs/reference/operators/add-custom-op.html，
"Wrapping an external inference runtime in a custom operator"）：

- 图中只有一个自定义域 ai.facehi 的节点 HighlightRemoval；
- 两个子模型（face_detector.onnx / face_landmarks_detector.onnx）、
  按 cli_process.load_cli_config 合并后的三种模式配置 YAML、
  Haar 兜底级联 XML，全部序列化进节点属性；
- 运行需要注册配套的自定义算子库 libfacehi_custom_ops.so（ORT 设计如此，
  任何含自定义域的 ONNX 都必须带 kernel 实现库）。

用法（需在完整仓库内运行，依赖 cli_process 与 onnx/models/ 子模型）：
    python git-high2/onnx/make_facehi_onnx.py --preset daily
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

_HERE = Path(__file__).resolve().parent
GH2_MODELS_DIR = _HERE.parent / "models"


def _find_repo_root() -> Path:
    for parent in (_HERE, *_HERE.parents):
        if (parent / "cli_process.py").is_file() and \
           (parent / "highlight_removal" / "pipeline.py").is_file():
            return parent
    raise SystemExit("必须在完整仓库内运行（未找到 cli_process.py / highlight_removal/）")


ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBMODELS_DIR = ROOT / "onnx" / "models"
MODES = ("常用模式", "高保真模式", "最高质量模式")
CONFIG_SECTIONS = ("face_detection", "regions", "highlight_detection",
                   "highlight_removal", "pipeline")

# 日常/平衡档的独立配置：每个数值键取「强力档 | 保护细节档」两端实际烘焙值
# 的中值（整数参数按 0.5 半进位取整），行尾注释标注两端来源值。
# 端点来自 facehi_strong.onnx（mode=强力模式）与 facehi_detail.onnx（mode=保护细节）
# 节点属性 config_yaml 中被烘焙选中的段。
DAILY_DETECTION = {
    "rgb_brightness_threshold": 205,        # 强力 192 | 细节 218
    "hsv_v_threshold": 190,                 # 178 | 202
    "hsv_s_upper": 150,                     # 158 | 142
    "lab_l_threshold": 188,                 # 178 | 198
    "local_brightness_threshold": 6,        # 3 | 8（5.5 半进位）
    "local_contrast_threshold": 0.026,      # 0.018 | 0.034
    "saturation_pixel_threshold": 244,      # 240 | 248
    "highlight_min_area": 10,               # 5 | 14（9.5 半进位）
    "highlight_max_area_ratio": 0.1225,     # 0.155 | 0.09
    "mask_dilate_radius": 1,                # 2 | 0
    "mask_erode_radius": 1,                 # 0 | 1（0.5 半进位）
    "mask_blur_radius": 9,                  # 8 | 10
    "oil_shine_s_upper": 175,               # 188 | 162
    "adaptive_region_delta": 5.1,           # 3.4 | 6.8
    "adaptive_core_delta": 8.3,             # 5.8 | 10.8
    "adaptive_grow_iterations": 12,         # 16 | 7（11.5 半进位）
    "morph_close_radius": 3,                # 4 | 2
    "soft_mask_gain": 1.635,                # 1.85 | 1.42
    "forehead_region_max_fraction": 0.285,  # 0.35 | 0.22
    "nose_tip_region_max_fraction": 0.55,   # 0.62 | 0.48
    "cheek_region_max_fraction": 0.20,      # 0.26 | 0.14
    "brow_region_max_fraction": 0.30,       # 0.32 | 0.28
}
DAILY_REMOVAL = {
    "mode": "混合",                             # 两端同为 混合
    "brightness_suppress_strength": 0.825,      # 0.97 | 0.68
    "chroma_restore_strength": 0.30,            # 0.42 | 0.18
    "texture_preserve_strength": 0.81,          # 0.62 | 1.0
    "edge_protect_strength": 0.46,              # 0.32 | 0.60
    "inpainting_radius": 5,                     # 7 | 3
    "poisson_alpha_strength": 0.68,             # 0.88 | 0.48
    "final_blend_alpha": 0.905,                 # 0.99 | 0.82
    "max_allowed_modify_area_ratio": 0.165,     # 0.24 | 0.09
    "max_allowed_mean_brightness_change": 18,   # 26 | 10
    "max_allowed_local_color_delta": 13,        # 18 | 8
    "faithful_luminance_floor": 0.675,          # 0.66 | 0.69
    "extreme_core_extra_l": 19,                 # 8 | 30
}

# 每个 preset 只改烘焙的 mode / 配置段，内核（自定义算子库）完全相同。
PRESETS = {
    "standard": {
        "filename": "facehi.onnx",
        "mode": "常用模式",
        "label": "常用（强力向）",
        "desc": "检测=灵敏、修复=强力、process_scale=compromise",
    },
    "daily": {
        "filename": "facehi_daily.onnx",
        "mode": "日常模式",
        "label": "日常/平衡",
        "desc": "强力/保护细节两端中值：检测阈值 rgb205/hsv_v190/lab_l188、grow 12、"
                "soft_mask_gain 1.635；修复压制 0.825、final_blend 0.905；"
                "method=混合、process_scale=compromise",
    },
}


def build_config_yaml(default_mode: str) -> str:
    """用 cli_process.load_cli_config 生成各模式合并后的配置（与 Python CLI 完全一致）。

    default_mode="日常模式" 时额外加入独立的日常/平衡段：以高保真模式的
    face_detection/regions/pipeline 为底座，高光检测与修复两段套用
    DAILY_DETECTION / DAILY_REMOVAL（强力档与保护细节档的逐项中值）。
    """
    from cli_process import load_cli_config

    modes = {}
    for mode in MODES:
        cfg = load_cli_config(mode)
        modes[mode] = {sec: cfg.get(sec, {}) for sec in CONFIG_SECTIONS}
    if default_mode == "日常模式":
        base = load_cli_config("高保真模式")
        daily = {sec: dict(base.get(sec, {})) for sec in CONFIG_SECTIONS}
        daily["highlight_detection"].update(DAILY_DETECTION)
        daily["highlight_removal"].update(DAILY_REMOVAL)
        modes["日常模式"] = daily
    doc = {"default_mode": default_mode, "modes": modes}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def bytes_tensor(name: str, data: bytes) -> onnx.TensorProto:
    arr = np.frombuffer(data, dtype=np.uint8)
    return helper.make_tensor(name, TensorProto.UINT8, [len(arr)], arr.tobytes(), raw=True)


def build_model(preset_name: str, out_path: Path) -> Path:
    preset = PRESETS[preset_name]
    detector = (SUBMODELS_DIR / "face_detector.onnx").read_bytes()
    landmarks = (SUBMODELS_DIR / "face_landmarks_detector.onnx").read_bytes()

    import cv2

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    haar_xml = haar_path.read_bytes() if haar_path.is_file() else b""

    config_yaml = build_config_yaml(default_mode=preset["mode"])

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
        mode=preset["mode"],
        doc_string=(
            "整条面部去高光流水线：BlazeFace 检测→478点关键点→分区→高光检测→修复→保护回写。"
            f"烘焙档位：{preset['label']}（mode={preset['mode']}，{preset['desc']}）"
        ),
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
            f"facehi 单一 ONNX 入口（{preset['label']}档，烘焙 mode={preset['mode']}）。"
            "输入 image：uint8 BGR [H,W,3]（与 cv2.imread 一致）；"
            "输出 result：uint8 BGR 同形状，highlight_mask：uint8 [H,W]。"
            "必须注册 libfacehi_custom_ops 自定义算子库后才能创建会话。"
        ),
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("ai.facehi", 1)],
        producer_name="facehi",
        doc_string=(
            "面部去高光整条流水线的单一 ONNX 封装（ORT 自定义算子包装外部流水线）。"
            f"档位：{preset['label']}，烘焙 mode={preset['mode']}（{preset['desc']}）"
        ),
    )
    model.ir_version = 9
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[完成] {out_path}（{size_mb:.2f} MB，preset={preset_name}，"
          f"烘焙 mode={preset['mode']}，内嵌 detector={len(detector)}B, "
          f"landmarks={len(landmarks)}B, haar={len(haar_xml)}B）")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preset",
        default="standard",
        choices=sorted(PRESETS),
        help="烘焙档位：standard=常用（强力向，facehi.onnx）；"
             "daily=日常/平衡（facehi_daily.onnx，烘焙 高保真模式）",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出路径（默认 git-high2/models/<preset 对应文件名>）",
    )
    args = parser.parse_args(argv)
    out = Path(args.output) if args.output else GH2_MODELS_DIR / PRESETS[args.preset]["filename"]
    build_model(args.preset, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
