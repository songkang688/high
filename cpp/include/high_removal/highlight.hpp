// 高光检测，对应 highlight_removal/highlight_detect.py。
#pragma once

#include "high_removal/regions.hpp"

namespace hr {

struct HighlightOutput {
    cv::Mat hard_mask;  // CV_8U 0/255
    cv::Mat soft_mask;  // CV_8U 0..255
    std::vector<std::string> warnings;
};

HighlightOutput detectHighlight(const cv::Mat& imageBgr, const RegionMasks& regions, const Section& params);

}  // namespace hr
