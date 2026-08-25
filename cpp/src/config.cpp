// YAML 配置加载。
#include "facehi/config.hpp"

#include <cstdlib>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace facehi {

namespace {

ParamValue scalar_to_value(const YAML::Node& node) {
  std::string raw = node.Scalar();
  std::string lowered = raw;
  for (auto& c : lowered) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lowered == "true") return true;
  if (lowered == "false") return false;
  // 完整数字才视为 double（避免 "0.25_0.5"、"compromise" 被误判）。
  if (!raw.empty()) {
    char* end = nullptr;
    double v = std::strtod(raw.c_str(), &end);
    if (end == raw.c_str() + raw.size()) return v;
  }
  return raw;
}

Params section_to_params(const YAML::Node& node) {
  Params p;
  if (!node || !node.IsMap()) return p;
  for (const auto& kv : node) {
    const std::string key = kv.first.as<std::string>();
    if (kv.second.IsScalar()) p.set(key, scalar_to_value(kv.second));
  }
  return p;
}

}  // namespace

Config load_config_yaml(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  Config cfg;
  cfg.face_detection = section_to_params(root["face_detection"]);
  cfg.regions = section_to_params(root["regions"]);
  cfg.highlight_detection = section_to_params(root["highlight_detection"]);
  cfg.highlight_removal = section_to_params(root["highlight_removal"]);
  cfg.pipeline = section_to_params(root["pipeline"]);
  return cfg;
}

namespace {

// 对应 cli_process._merge_preset。
void merge_preset(Params* base, const std::string& preset_path, const std::string& preset_name) {
  if (preset_name == "正常") return;
  YAML::Node root = YAML::LoadFile(preset_path);
  YAML::Node presets = root["presets"];
  if (!presets) return;
  YAML::Node preset = presets[preset_name];
  if (!preset) return;
  base->merge(section_to_params(preset));
}

}  // namespace

Config load_cli_config(const std::string& repo_root, const std::string& mode_name) {
  Config cfg = load_config_yaml(repo_root + "/configs/default.yaml");
  const std::string sensitivity = repo_root + "/configs/detection_sensitivity_presets.yaml";
  const std::string intensity = repo_root + "/configs/removal_intensity_presets.yaml";

  cfg.pipeline.set("enable_visualization", false);
  cfg.pipeline.set("enable_roi_removal", true);
  cfg.pipeline.set("roi_padding_ratio", 0.12);
  cfg.pipeline.set("refine_upsampled_masks", false);

  if (mode_name == "常用模式") {
    cfg.pipeline.set("process_scale", std::string("compromise"));
    merge_preset(&cfg.highlight_detection, sensitivity, "灵敏");
    merge_preset(&cfg.highlight_removal, intensity, "强力");
  } else if (mode_name == "高保真模式") {
    cfg.pipeline.set("process_scale", std::string("compromise"));
  } else if (mode_name == "最高质量模式") {
    cfg.pipeline.set("process_scale", 1.0);
  } else {
    throw std::runtime_error("未知处理模式: " + mode_name);
  }
  return cfg;
}

}  // namespace facehi
