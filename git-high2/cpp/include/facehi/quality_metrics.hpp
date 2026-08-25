// 对应 highlight_removal/quality_metrics.py。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"
#include "facehi/face_regions.hpp"

namespace facehi {

// 一份质量审计指标（键名与 Python 输出保持一致的语义）。
struct Metrics {
  double face_confidence = 0.0;        // 人脸检测置信度
  int landmark_count = 0;              // 关键点数量
  cv::Rect face_bbox;                  // 人脸 bounding box
  double highlight_area_ratio = 0.0;   // 高光 mask 面积占人脸比例
  double modified_area_ratio = 0.0;    // 实际修改像素占人脸比例
  double mean_brightness_change = 0.0; // 平均亮度变化
  double max_brightness_change = 0.0;  // 最大亮度变化
  double mean_color_change = 0.0;      // 平均色差变化
  double max_color_change = 0.0;       // 最大色差变化
  bool eyes_modified = false;          // 眼睛区域是否被修改
  bool lips_modified = false;          // 嘴唇区域是否被修改
  bool brows_modified = false;         // 眉毛区域是否被修改
  bool background_modified = false;    // 背景是否被修改
  double mean_da = 0.0;                // 平均 a 通道变化
  double mean_db = 0.0;                // 平均 b 通道变化
  bool has_warnings = false;           // 是否触发真实性警告
};

struct FaceInfo {
  double confidence = 0.0;
  int landmark_count = 0;
  cv::Rect bbox;
};

Metrics compute_metrics(const cv::Mat& original_bgr, const cv::Mat& result_bgr,
                        const cv::Mat& hard_mask, const RegionMasks& regions,
                        const FaceInfo& face, const Params& params,
                        std::vector<std::string>* warnings);

std::string metrics_to_text(const Metrics& metrics, const std::vector<std::string>& warnings);

}  // namespace facehi
