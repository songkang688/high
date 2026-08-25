// 对应 highlight_removal/remove_highlight.py。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"
#include "facehi/face_regions.hpp"

namespace facehi {

struct RemovalOutput {
  cv::Mat result_bgr;
  cv::Mat applied_mask;
  cv::Mat diff_bgr;
  std::vector<std::string> warnings;
};

RemovalOutput remove_highlight(const cv::Mat& image_bgr, const cv::Mat& hard_mask,
                               const cv::Mat& soft_mask, const RegionMasks& regions,
                               const Params& params);

}  // namespace facehi
