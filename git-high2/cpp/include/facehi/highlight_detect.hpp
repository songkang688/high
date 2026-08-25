// 对应 highlight_removal/highlight_detect.py。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"
#include "facehi/face_regions.hpp"

namespace facehi {

struct HighlightOutput {
  cv::Mat hard_mask;  // CV_8U 0/255
  cv::Mat soft_mask;  // CV_8U 0..255
  std::vector<std::string> warnings;
};

HighlightOutput detect_highlight(const cv::Mat& image_bgr, const RegionMasks& regions,
                                 const Params& params);

}  // namespace facehi
