// 对应 highlight_removal/face_regions.py 与 skin_mask.py。
#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"
#include "facehi/face_landmarks.hpp"

namespace facehi {

// 对应 RegionMasks dataclass。
struct RegionMasks {
  std::map<std::string, cv::Mat> masks;  // 全部为 CV_8U 0/255（soft 除外）
  FaceGeometry geometry;
  std::vector<std::string> warnings;
  std::vector<cv::Point2f> face_contour_pts;  // Python 中存于 masks["face_contour_pts"]
};

// 对应 skin_mask.estimate_skin_mask。
cv::Mat estimate_skin_mask(const cv::Mat& image_bgr, const cv::Mat& face_mask,
                           const cv::Mat& protect_mask, const Params& params);

// 对应 face_regions.build_face_regions（含 simple_protect_mode 与完整模式）。
RegionMasks build_face_regions(const cv::Mat& image_bgr, const cv::Mat& landmarks,
                               const Params& params);

}  // namespace facehi
