// 主流程编排，对应 highlight_removal/pipeline.py 的推理路径
//（enable_alignment=false、refine_upsampled_masks=false，即 configs/default.yaml 默认路径）。
#pragma once

#include "high_removal/highlight.hpp"
#include "high_removal/landmarker.hpp"
#include "high_removal/removal.hpp"

namespace hr {

struct PipelineOutput {
    bool success = false;
    std::string status;
    std::vector<std::string> warnings;
    cv::Mat result_bgr;
    cv::Mat highlight_mask;
    cv::Mat soft_mask;
};

PipelineOutput processImage(const cv::Mat& imageBgr, const Config& config, const FaceLandmarkerOrt& landmarker);

}  // namespace hr
