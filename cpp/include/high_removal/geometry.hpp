// 人脸几何量，对应 highlight_removal/face_landmarks.py。
#pragma once

#include "high_removal/common.hpp"

namespace hr {

struct FaceGeometry {
    cv::Point2f left_eye_center;
    cv::Point2f right_eye_center;
    cv::Point2f eye_mid;
    cv::Point2f nose_tip;
    cv::Point2f mouth_left;
    cv::Point2f mouth_right;
    cv::Point2f chin;
    float interocular = 0.f;
    Bbox face_bbox;
    cv::Vec2f x_axis{1.f, 0.f};
    cv::Vec2f y_axis{0.f, 1.f};
    float angle_deg = 0.f;
};

// pts()：取有效索引处的 (x, y)。
std::vector<cv::Point2f> pickPts(const Landmarks& lm, const std::vector<int>& indices);
cv::Point2f centerOf(const Landmarks& lm, const std::vector<int>& indices);
FaceGeometry getGeometry(const Landmarks& lm, cv::Size imageSize);
Bbox bboxFromLandmarks(const Landmarks& lm, cv::Size imageSize);
double estimateHeadYawRatio(const Landmarks& lm);
// _filter_side_brow_pts()
std::vector<cv::Point2f> filterSideBrowPts(const std::vector<cv::Point2f>& brow, const std::vector<cv::Point2f>& eye, double io);
// project_points()：投影到局部坐标 (u, v)。
void projectPoints(const std::vector<cv::Point2f>& p, cv::Point2f origin, cv::Vec2f xAxis, cv::Vec2f yAxis,
                   std::vector<float>& u, std::vector<float>& v);

}  // namespace hr
