// YAML 配置加载：与 configs/default.yaml、cli_process.load_cli_config 语义一致。
#pragma once

#include <string>

#include "facehi/common.hpp"

namespace facehi {

// 读取 default.yaml 的各段。
Config load_config_yaml(const std::string& path);

// 对应 cli_process.load_cli_config：常用模式 / 高保真模式 / 最高质量模式。
// repo_root：仓库根目录（用于 configs/*.yaml）。
Config load_cli_config(const std::string& repo_root, const std::string& mode_name);

// 从内嵌 YAML 文本加载（facehi.onnx 的 config_yaml 属性）。
// 文本格式（由 onnx/make_facehi_onnx.py 生成，各模式已按 cli_process.load_cli_config 合并完毕）：
//   default_mode: 常用模式
//   modes:
//     常用模式:   {face_detection: {...}, regions: {...}, ...}
//     高保真模式: {...}
//     最高质量模式: {...}
// mode_name 为空时使用 default_mode。
Config load_embedded_config(const std::string& yaml_text, const std::string& mode_name);

}  // namespace facehi
