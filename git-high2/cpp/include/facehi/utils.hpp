// 对应 highlight_removal/utils.py 的图像与 mask 工具。
#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"

namespace facehi {

// mask 归一化：uint8 → float32 [0,1]。
cv::Mat normalize_mask(const cv::Mat& mask_u8);

// 形态学：op ∈ {dilate, erode, open, close}，椭圆核半径 radius（Python round 语义）。
cv::Mat morph(const cv::Mat& mask_u8, const std::string& op, double radius);

// 对应 blur_mask：ksize = 2*round(r)+1 的高斯。
cv::Mat blur_mask(const cv::Mat& mask_u8, double radius);

// 多边形/凸包/折线/椭圆 mask（与 utils.py 相同的取整与线型）。
cv::Mat fill_poly_mask(const cv::Size& size, const std::vector<cv::Point2f>& points);
cv::Mat fill_hull_mask(const cv::Size& size, const std::vector<cv::Point2f>& points);
cv::Mat line_mask(const cv::Size& size, const std::vector<cv::Point2f>& points, int thickness);
cv::Mat ellipse_mask(const cv::Size& size, const cv::Point2f& center, const cv::Size2d& axes, double angle);

// 处理比例。
double clamp_process_scale(double scale);
// 返回 (face_scale, highlight_scale, is_compromise)。
std::tuple<double, double, bool> resolve_process_scales(const Params& pipeline);

cv::Mat resize_for_process(const cv::Mat& image, double scale);
cv::Mat scale_landmarks(const cv::Mat& landmarks, double scale);

cv::Mat upsample_mask_hard(const cv::Mat& mask_u8, const cv::Size& size);
cv::Mat upsample_mask_soft(const cv::Mat& mask_u8, const cv::Size& size);

// 低分辨率检测时缩放阈值类参数。
Params scaled_highlight_params(const Params& params, double scale);

// ROI 计算（hard|soft 的活动区域 + padding）。
std::optional<cv::Rect> compute_roi_bbox(const cv::Mat& hard_u8, const cv::Mat& soft_u8,
                                         const cv::Size& size, double padding_ratio);

// 图像读写（字节流方式，兼容中文路径）。
cv::Mat read_image_bgr(const std::string& path);
bool write_image(const std::string& path, const cv::Mat& image_bgr);

}  // namespace facehi
