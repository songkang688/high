// 区域划分与肤色掩码，对应 highlight_removal/face_regions.py 与 skin_mask.py。
#pragma once

#include "high_removal/common.hpp"
#include "high_removal/geometry.hpp"

namespace hr {

struct RegionMasks {
    std::map<std::string, cv::Mat> masks;  // CV_8U，0/255
    FaceGeometry geometry;
    std::vector<std::string> warnings;

    bool has(const std::string& key) const { return masks.count(key) > 0; }
    const cv::Mat& at(const std::string& key) const { return masks.at(key); }
};

// estimate_skin_mask()
cv::Mat estimateSkinMask(const cv::Mat& imageBgr, const cv::Mat& faceMask, const cv::Mat& protectMask, int smoothRadius);

// build_face_regions()：simple_protect_mode 与 key_region_detection_mode 两条路径均支持。
RegionMasks buildFaceRegions(const cv::Mat& imageBgr, const Landmarks& lm, const Section& params);

}  // namespace hr
