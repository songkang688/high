// 去高光修复，对应 highlight_removal/remove_highlight.py。
#pragma once

#include "high_removal/regions.hpp"

namespace hr {

struct RemovalOutput {
    cv::Mat result_bgr;
    std::vector<std::string> warnings;
};

// params 为合并后的参数（highlight_removal ∪ highlight_detection，detection 覆盖同名键，
// 另加 enable_roi_removal / roi_padding_ratio），对应 pipeline._merge_removal_params()。
RemovalOutput removeHighlight(const cv::Mat& imageBgr, const cv::Mat& hardMask, const cv::Mat& softMask,
                              const RegionMasks& regions, const Section& params);

}  // namespace hr
