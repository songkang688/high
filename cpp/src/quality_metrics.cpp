// 对应 highlight_removal/quality_metrics.py。
#include "facehi/quality_metrics.hpp"

#include <cmath>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace facehi {

namespace {

double round_n(double v, int digits) {
  double f = std::pow(10.0, digits);
  return std::nearbyint(v * f) / f;  // Python round：half-to-even
}

bool modified_in(const cv::Mat& mask_u8, const cv::Mat& diff_gray, int threshold = 2) {
  for (int y = 0; y < mask_u8.rows; ++y) {
    const uint8_t* m = mask_u8.ptr<uint8_t>(y);
    const uint8_t* d = diff_gray.ptr<uint8_t>(y);
    for (int x = 0; x < mask_u8.cols; ++x)
      if (m[x] > 0 && d[x] > threshold) return true;
  }
  return false;
}

}  // namespace

Metrics compute_metrics(const cv::Mat& original_bgr, const cv::Mat& result_bgr,
                        const cv::Mat& hard_mask, const RegionMasks& regions,
                        const FaceInfo& face, const Params& params,
                        std::vector<std::string>* warnings) {
  Metrics m;
  const int h = original_bgr.rows, w = original_bgr.cols;
  const cv::Mat& face_mask = regions.masks.at("face_mask");

  cv::Mat diff, diff_gray;
  cv::absdiff(original_bgr, result_bgr, diff);
  cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);

  cv::Mat lab0_u8, lab1_u8;
  cv::cvtColor(original_bgr, lab0_u8, cv::COLOR_BGR2Lab);
  cv::cvtColor(result_bgr, lab1_u8, cv::COLOR_BGR2Lab);

  long long face_count = 0, hard_count = 0, mod_face_count = 0;
  bool background_modified = false;
  double sum_abs_dl = 0.0, max_abs_dl = 0.0, sum_de = 0.0, max_de = 0.0;
  double sum_da = 0.0, sum_db = 0.0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* fm = face_mask.ptr<uint8_t>(y);
    const uint8_t* hm = hard_mask.ptr<uint8_t>(y);
    const uint8_t* dg = diff_gray.ptr<uint8_t>(y);
    const cv::Vec3b* l0 = lab0_u8.ptr<cv::Vec3b>(y);
    const cv::Vec3b* l1 = lab1_u8.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      bool in_face = fm[x] > 0;
      bool changed = dg[x] > 2;
      face_count += in_face ? 1 : 0;
      hard_count += hm[x] > 0 ? 1 : 0;
      if (changed && !in_face) background_modified = true;
      if (changed && in_face) {
        ++mod_face_count;
        float dl = static_cast<float>(l1[x][0]) - static_cast<float>(l0[x][0]);
        float da = static_cast<float>(l1[x][1]) - static_cast<float>(l0[x][1]);
        float db = static_cast<float>(l1[x][2]) - static_cast<float>(l0[x][2]);
        double de = std::sqrt(static_cast<double>(dl) * dl + static_cast<double>(da) * da +
                              static_cast<double>(db) * db);
        sum_abs_dl += std::abs(dl);
        max_abs_dl = std::max(max_abs_dl, static_cast<double>(std::abs(dl)));
        sum_de += de;
        max_de = std::max(max_de, de);
        sum_da += da;
        sum_db += db;
      }
    }
  }

  m.face_confidence = round_n(face.confidence, 4);
  m.landmark_count = face.landmark_count;
  m.face_bbox = face.bbox;
  m.highlight_area_ratio = round_n(static_cast<double>(hard_count) / std::max<long long>(1, face_count), 6);
  m.modified_area_ratio = round_n(static_cast<double>(mod_face_count) / std::max<long long>(1, face_count), 6);
  m.mean_brightness_change = mod_face_count ? round_n(sum_abs_dl / mod_face_count, 4) : 0.0;
  m.max_brightness_change = mod_face_count ? round_n(max_abs_dl, 4) : 0.0;
  m.mean_color_change = mod_face_count ? round_n(sum_de / mod_face_count, 4) : 0.0;
  m.max_color_change = mod_face_count ? round_n(max_de, 4) : 0.0;

  cv::Mat eye_union;
  cv::bitwise_or(regions.masks.at("left_eye"), regions.masks.at("right_eye"), eye_union);
  cv::Mat brow_union;
  cv::bitwise_or(regions.masks.at("left_brow_protect"), regions.masks.at("right_brow_protect"), brow_union);
  m.eyes_modified = modified_in(eye_union, diff_gray);
  m.lips_modified = modified_in(regions.masks.at("lips"), diff_gray);
  m.brows_modified = modified_in(brow_union, diff_gray);
  m.background_modified = background_modified;

  double max_area = params.getd("max_allowed_modify_area_ratio", 0.15);
  double max_mean_L = params.getd("max_allowed_mean_brightness_change", 18);
  double max_mean_E = params.getd("max_allowed_local_color_delta", 18);
  if (m.modified_area_ratio > max_area)
    warnings->push_back("修改区域超过人脸皮肤区域的 15% 或当前设置上限");
  if (m.eyes_modified) warnings->push_back("眼睛区域被修改，已触发真实性警告");
  if (m.lips_modified) warnings->push_back("嘴唇区域被修改，已触发真实性警告");
  if (m.brows_modified) warnings->push_back("眉毛区域被修改，已触发真实性警告");
  if (m.background_modified) warnings->push_back("背景区域被修改，已触发真实性警告");
  if (m.mean_brightness_change > max_mean_L)
    warnings->push_back("平均亮度变化过大，可能影响真实性，请降低亮度压制强度");
  if (m.mean_color_change > max_mean_E)
    warnings->push_back("平均色差变化过大，可能导致肤色失真，请降低色度恢复或融合强度");

  m.mean_da = mod_face_count ? round_n(sum_da / mod_face_count, 4) : 0.0;
  m.mean_db = mod_face_count ? round_n(sum_db / mod_face_count, 4) : 0.0;
  if (std::abs(m.mean_da) > 6 || std::abs(m.mean_db) > 8)
    warnings->push_back("处理后肤色可能偏灰、偏红或偏黄，请降低色度恢复强度");

  m.has_warnings = !warnings->empty();
  return m;
}

std::string metrics_to_text(const Metrics& m, const std::vector<std::string>& warnings) {
  std::ostringstream os;
  os << "人脸检测置信度：" << m.face_confidence << "\n";
  os << "关键点数量：" << m.landmark_count << "\n";
  os << "人脸 bounding box 坐标：[" << m.face_bbox.x << ", " << m.face_bbox.y << ", "
     << m.face_bbox.width << ", " << m.face_bbox.height << "]\n";
  os << "高光 mask 面积占人脸比例：" << m.highlight_area_ratio << "\n";
  os << "实际修改像素占人脸比例：" << m.modified_area_ratio << "\n";
  os << "平均亮度变化：" << m.mean_brightness_change << "\n";
  os << "最大亮度变化：" << m.max_brightness_change << "\n";
  os << "平均色差变化：" << m.mean_color_change << "\n";
  os << "最大色差变化：" << m.max_color_change << "\n";
  os << "眼睛区域是否被修改：" << (m.eyes_modified ? "True" : "False") << "\n";
  os << "嘴唇区域是否被修改：" << (m.lips_modified ? "True" : "False") << "\n";
  os << "眉毛区域是否被修改：" << (m.brows_modified ? "True" : "False") << "\n";
  os << "背景是否被修改：" << (m.background_modified ? "True" : "False") << "\n";
  os << "平均 a 通道变化：" << m.mean_da << "\n";
  os << "平均 b 通道变化：" << m.mean_db << "\n";
  os << "是否触发真实性警告：" << (m.has_warnings ? "True" : "False") << "\n";
  if (!warnings.empty()) {
    os << "\n真实性警告：\n";
    for (const auto& w : warnings) os << "- " << w << "\n";
  } else {
    os << "\n真实性警告：无\n";
  }
  return os.str();
}

}  // namespace facehi
