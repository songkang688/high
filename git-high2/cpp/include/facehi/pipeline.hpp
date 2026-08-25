// 对应 highlight_removal/pipeline.py。
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"
#include "facehi/face_detect.hpp"
#include "facehi/face_regions.hpp"
#include "facehi/quality_metrics.hpp"

namespace facehi {

// 对应 PipelineOutput（不含可视化视图；CLI 模式下 Python 也关闭可视化）。
struct PipelineOutput {
  bool success = false;
  std::string status;
  std::vector<std::string> warnings;
  cv::Mat result_bgr;
  cv::Mat highlight_mask;  // hard mask
  cv::Mat soft_mask;
  cv::Mat diff_bgr;
  Metrics metrics;
  std::string metrics_text;
  std::optional<RegionMasks> regions;
  std::optional<FaceResult> face;
  int detection_count = 0;
};

// 单入口 API：一次调用完成 检测 → 分区 → 高光 → 修复 → 质量审计。
// landmarker 可为 nullptr（此时仅在提供 injected_landmarks 时可用）。
PipelineOutput process_image(const cv::Mat& image_bgr, const Config& config,
                             const OnnxFaceLandmarker* landmarker,
                             const cv::Mat* injected_landmarks = nullptr);

}  // namespace facehi
