#include "high_removal/common.hpp"

#include <yaml-cpp/yaml.h>

namespace hr {

static Value parseScalar(const YAML::Node& node) {
    // 与 PyYAML safe_load 的标量解析保持一致：bool → 数字 → 字符串。
    try {
        const bool b = node.as<bool>();
        const std::string raw = node.Scalar();
        // yaml-cpp 会把 "y"/"n" 等也当 bool；限制到 PyYAML 认可的字面量。
        static const char* kBoolLiterals[] = {"true", "false", "True", "False", "TRUE", "FALSE",
                                              "yes", "no", "Yes", "No", "YES", "NO",
                                              "on", "off", "On", "Off", "ON", "OFF"};
        for (const char* lit : kBoolLiterals) {
            if (raw == lit) return b;
        }
    } catch (...) {
    }
    try {
        return node.as<double>();
    } catch (...) {
    }
    return node.as<std::string>();
}

static Section parseSection(const YAML::Node& node) {
    Section out;
    if (!node || !node.IsMap()) return out;
    for (const auto& kv : node) {
        if (kv.second.IsScalar()) {
            out[kv.first.as<std::string>()] = parseScalar(kv.second);
        }
    }
    return out;
}

Config loadConfig(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);
    Config cfg;
    cfg.face_detection = parseSection(root["face_detection"]);
    cfg.regions = parseSection(root["regions"]);
    cfg.highlight_detection = parseSection(root["highlight_detection"]);
    cfg.highlight_removal = parseSection(root["highlight_removal"]);
    cfg.pipeline = parseSection(root["pipeline"]);
    return cfg;
}

double getF(const Section& s, std::initializer_list<const char*> names, double def) {
    for (const char* name : names) {
        auto it = s.find(name);
        if (it == s.end()) continue;
        if (std::holds_alternative<double>(it->second)) return std::get<double>(it->second);
        if (std::holds_alternative<bool>(it->second)) return std::get<bool>(it->second) ? 1.0 : 0.0;
        return std::stod(std::get<std::string>(it->second));
    }
    return def;
}

bool getB(const Section& s, std::initializer_list<const char*> names, bool def) {
    for (const char* name : names) {
        auto it = s.find(name);
        if (it == s.end()) continue;
        if (std::holds_alternative<bool>(it->second)) return std::get<bool>(it->second);
        if (std::holds_alternative<double>(it->second)) return std::get<double>(it->second) != 0.0;
        return !std::get<std::string>(it->second).empty();
    }
    return def;
}

std::string getS(const Section& s, std::initializer_list<const char*> names, const std::string& def) {
    for (const char* name : names) {
        auto it = s.find(name);
        if (it == s.end()) continue;
        if (std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
        if (std::holds_alternative<bool>(it->second)) return std::get<bool>(it->second) ? "true" : "false";
        return std::to_string(std::get<double>(it->second));
    }
    return def;
}

}  // namespace hr
