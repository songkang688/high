"""生成单文件模型 models/high_removal.onnx。

按微软官方 create_custom_op_wrapper.py 的做法
（https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/custom_op_wrapper/create_custom_op_wrapper.py）：
图中只有一个自定义算子节点 ai.high:HighlightRemoval，
两个人脸 ONNX 模型与 configs/default.yaml 的全部字节序列化为节点属性（uint8 张量），
运行时由 libhigh_removal_ops.so 中的内核反序列化并执行 C++ 去高光流水线。

用户因此只需要携带两个文件：
  1. models/high_removal.onnx      —— 本脚本产物（含全部权重与配置）
  2. libhigh_removal_ops.so        —— 自定义算子内核（cpp/ 构建产物）

用法：
  python tools/export_high_removal_onnx.py            # 使用仓库默认路径
  python tools/export_high_removal_onnx.py --output models/high_removal.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]

DOMAIN = "ai.high"
OP_NAME = "HighlightRemoval"
OPSET_VERSION = 1
# ORT 1.22（C++ 侧最低要求）支持到 IR v10，显式固定避免 onnx 包升级导致模型不可加载。
IR_VERSION = 10


def bytes_attr(name: str, path: Path) -> onnx.TensorProto:
    data = path.read_bytes()
    if not data:
        raise SystemExit(f"[错误] 属性文件为空: {path}")
    return helper.make_tensor(name=name, data_type=TensorProto.UINT8, dims=[len(data)], vals=data, raw=True)


def build_model(detector: Path, landmarks: Path, config_yaml: Path) -> onnx.ModelProto:
    node = helper.make_node(
        OP_NAME,
        name=f"{OP_NAME}_0",
        inputs=["image"],
        outputs=["result", "hard_mask"],
        domain=DOMAIN,
        # 属性名与 cpp/src/custom_op.cpp 中的 kAttr* 常量一一对应。
        face_detector_model=bytes_attr("face_detector_model", detector),
        face_landmarks_model=bytes_attr("face_landmarks_model", landmarks),
        config_yaml=bytes_attr("config_yaml", config_yaml),
    )

    # 动态 H/W（dim_param），通道固定 3；BGR uint8，与 cv2.imread 直接对接。
    image = helper.make_tensor_value_info("image", TensorProto.UINT8, ["H", "W", 3])
    result = helper.make_tensor_value_info("result", TensorProto.UINT8, ["H", "W", 3])
    hard_mask = helper.make_tensor_value_info("hard_mask", TensorProto.UINT8, ["H", "W"])

    graph = helper.make_graph([node], "high_removal", [image], [result, hard_mask])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(DOMAIN, OPSET_VERSION)],
        producer_name="high_removal.export_high_removal_onnx",
        doc_string=(
            "证件照去高光单文件模型：单个 ai.high:HighlightRemoval 自定义算子封装完整 C++ 流水线"
            "（MediaPipe 等价关键点 + OpenCV 高光检测/修复）。"
            "运行前需通过 register_custom_ops_library 加载 libhigh_removal_ops.so。"
            "输入 image: uint8 [H,W,3] BGR；输出 result: uint8 [H,W,3] BGR，hard_mask: uint8 [H,W]。"
        ),
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", default=str(ROOT / "models" / "face_detector.onnx"))
    parser.add_argument("--landmarks", default=str(ROOT / "models" / "face_landmarks_detector.onnx"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output", "-o", default=str(ROOT / "models" / "high_removal.onnx"))
    args = parser.parse_args()

    for p in (args.detector, args.landmarks, args.config):
        if not Path(p).is_file():
            raise SystemExit(f"[错误] 找不到输入文件: {p}")

    model = build_model(Path(args.detector), Path(args.landmarks), Path(args.config))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"[信息] 已生成 {out}（{out.stat().st_size / 1024 / 1024:.2f} MiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
