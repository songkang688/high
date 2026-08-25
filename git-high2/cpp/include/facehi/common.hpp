// 通用类型与数值工具：参数表、Python 兼容取整、numpy 兼容分位数。
// 数值语义严格对齐 Python/numpy 版本（highlight_removal 包）。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

namespace facehi {

// ---------------------------------------------------------------------------
// 参数表：对应 Python dict 的一段配置（face_detection / regions / ...）。
// ---------------------------------------------------------------------------
using ParamValue = std::variant<double, bool, std::string>;

class Params {
 public:
  void set(const std::string& key, ParamValue v) { kv_[key] = std::move(v); }
  bool has(const std::string& key) const { return kv_.count(key) > 0; }

  // 对应 highlight_detect._p：按别名列表依次取值。
  double getd(const std::vector<std::string>& names, double def) const {
    for (const auto& n : names) {
      auto it = kv_.find(n);
      if (it == kv_.end()) continue;
      if (const double* d = std::get_if<double>(&it->second)) return *d;
      if (const bool* b = std::get_if<bool>(&it->second)) return *b ? 1.0 : 0.0;
    }
    return def;
  }
  // 用 const char* 承接单键调用，避免 {"a","b"} 花括号列表与
  // std::string(first,last) 迭代器构造产生二义性。
  double getd(const char* name, double def) const {
    return getd(std::vector<std::string>{std::string(name)}, def);
  }
  int geti(const char* name, int def) const {
    return static_cast<int>(getd(name, static_cast<double>(def)));
  }
  int geti(const std::vector<std::string>& names, int def) const {
    return static_cast<int>(getd(names, static_cast<double>(def)));
  }
  bool getb(const std::string& name, bool def) const {
    auto it = kv_.find(name);
    if (it == kv_.end()) return def;
    if (const bool* b = std::get_if<bool>(&it->second)) return *b;
    if (const double* d = std::get_if<double>(&it->second)) return *d != 0.0;
    return def;
  }
  std::string gets(const std::string& name, const std::string& def) const {
    auto it = kv_.find(name);
    if (it == kv_.end()) return def;
    if (const std::string* s = std::get_if<std::string>(&it->second)) return *s;
    if (const double* d = std::get_if<double>(&it->second)) {
      double v = *d;
      if (v == static_cast<long long>(v)) return std::to_string(static_cast<long long>(v));
      return std::to_string(v);
    }
    return def;
  }
  const std::map<std::string, ParamValue>& raw() const { return kv_; }
  void merge(const Params& other) {
    for (const auto& [k, v] : other.kv_) kv_[k] = v;
  }

 private:
  std::map<std::string, ParamValue> kv_;
};

// 完整配置：对应 default.yaml 的各段。
struct Config {
  Params face_detection;
  Params regions;
  Params highlight_detection;
  Params highlight_removal;
  Params pipeline;
};

// ---------------------------------------------------------------------------
// Python 语义取整
// ---------------------------------------------------------------------------

// Python round()：half-to-even（银行家舍入），对应 int(round(x))。
inline long long pyround(double x) {
  double r = std::nearbyint(x);  // 默认 FE_TONEAREST 即 half-to-even
  return static_cast<long long>(r);
}

// ---------------------------------------------------------------------------
// numpy 兼容分位数（linear 插值，含 numpy._lerp 的 gamma>=0.5 对称分支）
// ---------------------------------------------------------------------------
inline double np_percentile(std::vector<float>& vals, double q) {
  const size_t n = vals.size();
  if (n == 0) return 0.0;
  if (n == 1) return static_cast<double>(vals[0]);
  std::sort(vals.begin(), vals.end());
  double virtual_index = (q / 100.0) * static_cast<double>(n - 1);
  double lo = std::floor(virtual_index);
  double gamma = virtual_index - lo;
  size_t i = static_cast<size_t>(lo);
  size_t j = std::min(i + 1, n - 1);
  double a = static_cast<double>(vals[i]);
  double b = static_cast<double>(vals[j]);
  if (gamma >= 0.5) return b - (b - a) * (1.0 - gamma);
  return a + (b - a) * gamma;
}

// 对应 highlight_detect._percentile：空集合回退默认值。
inline double percentile_or(std::vector<float>& vals, double q, double fallback) {
  if (vals.empty()) return fallback;
  return np_percentile(vals, q);
}

inline double np_median(std::vector<float>& vals, double fallback) {
  if (vals.empty()) return fallback;
  return np_percentile(vals, 50.0);
}

// 从 float 图中收集 mask 内像素。
inline std::vector<float> masked_values(const cv::Mat& img32f, const cv::Mat& mask_bool8) {
  CV_Assert(img32f.type() == CV_32F && mask_bool8.type() == CV_8U);
  std::vector<float> out;
  out.reserve(1024);
  for (int y = 0; y < img32f.rows; ++y) {
    const float* p = img32f.ptr<float>(y);
    const uint8_t* m = mask_bool8.ptr<uint8_t>(y);
    for (int x = 0; x < img32f.cols; ++x)
      if (m[x]) out.push_back(p[x]);
  }
  return out;
}

}  // namespace facehi
