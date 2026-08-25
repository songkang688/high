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

}  // namespace facehi
