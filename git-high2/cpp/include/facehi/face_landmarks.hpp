// MediaPipe FaceMesh 468/478 语义索引与人脸几何。逐字对应 face_landmarks.py。
#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace facehi {

// 与 face_landmarks.py 完全一致的索引常量。
extern const std::vector<int> FACE_OVAL;
extern const std::vector<int> LEFT_EYE;
extern const std::vector<int> RIGHT_EYE;
extern const std::vector<int> LEFT_BROW;
extern const std::vector<int> RIGHT_BROW;
extern const std::vector<int> OUTER_LIPS;
extern const std::vector<int> INNER_LIPS;
extern const std::vector<int> NOSE_BRIDGE;
extern const std::vector<int> NOSE_TIP;
extern const std::vector<int> NOSE_WING;
extern const std::vector<int> MOUTH_CORNERS;
extern const std::vector<int> CHIN_AREA;
extern const std::vector<int> FOREHEAD_SIDE;
extern const std::vector<int> LEFT_LOWER_EYE;
extern const std::vector<int> RIGHT_LOWER_EYE;

// 对应 FaceGeometry dataclass。
struct FaceGeometry {
  cv::Point2f left_eye_center;
  cv::Point2f right_eye_center;
  cv::Point2f eye_mid;
  cv::Point2f nose_tip;
  cv::Point2f mouth_left;
  cv::Point2f mouth_right;
  cv::Point2f chin;
  double interocular = 0.0;
  cv::Rect face_bbox;
  cv::Vec2f x_axis;
  cv::Vec2f y_axis;
  double angle_deg = 0.0;
};

// landmarks: N x 3 CV_32F（x, y, z），允许 NaN 表示无效点。

// 对应 face_landmarks.pts：取有效点的 (x, y)。
std::vector<cv::Point2f> lm_pts(const cv::Mat& landmarks, const std::vector<int>& indices);

// 对应 center_of。
cv::Point2f lm_center_of(const cv::Mat& landmarks, const std::vector<int>& indices);

// 对应 estimate_head_yaw_ratio。
double estimate_head_yaw_ratio(const cv::Mat& landmarks);

// 对应 _filter_side_brow_pts。
std::vector<cv::Point2f> filter_side_brow_pts(const std::vector<cv::Point2f>& brow_pts,
                                              const std::vector<cv::Point2f>& eye_pts,
                                              double io);

// 对应 get_geometry。
FaceGeometry get_geometry(const cv::Mat& landmarks, const cv::Size& image_size);

// 对应 bbox_from_landmarks，返回 (x, y, w, h)。
cv::Rect bbox_from_landmarks(const cv::Mat& landmarks, const cv::Size& image_size);

// 对应 project_points：把点投影到 (origin, x_axis, y_axis) 局部坐标。
void project_points(const std::vector<cv::Point2f>& points, const cv::Point2f& origin,
                    const cv::Vec2f& x_axis, const cv::Vec2f& y_axis,
                    std::vector<float>* u, std::vector<float>* v);

}  // namespace facehi
