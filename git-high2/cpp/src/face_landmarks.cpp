// 对应 highlight_removal/face_landmarks.py。
#include "facehi/face_landmarks.hpp"

#include <cmath>

#include "facehi/common.hpp"

namespace facehi {

const std::vector<int> FACE_OVAL = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
};
const std::vector<int> LEFT_EYE = {33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7};
const std::vector<int> RIGHT_EYE = {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382};
const std::vector<int> LEFT_BROW = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};
const std::vector<int> RIGHT_BROW = {336, 296, 334, 293, 300, 285, 295, 282, 283, 276};
const std::vector<int> OUTER_LIPS = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185};
const std::vector<int> INNER_LIPS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191};
const std::vector<int> NOSE_BRIDGE = {168, 6, 197, 195, 5, 4};
const std::vector<int> NOSE_TIP = {1, 4, 5, 19, 94, 195, 197, 2};
const std::vector<int> NOSE_WING = {49, 98, 97, 2, 326, 327, 279};
const std::vector<int> MOUTH_CORNERS = {61, 291};
const std::vector<int> CHIN_AREA = {152, 148, 176, 149, 150, 136, 172, 58, 172, 378, 400, 377, 152, 365, 379};
const std::vector<int> FOREHEAD_SIDE = {54, 103, 67, 109, 10, 338, 297, 332, 284};
const std::vector<int> LEFT_LOWER_EYE = {33, 7, 163, 144, 145, 153, 154, 155, 133};
const std::vector<int> RIGHT_LOWER_EYE = {362, 382, 381, 380, 374, 373, 390, 249, 263};

std::vector<cv::Point2f> lm_pts(const cv::Mat& landmarks, const std::vector<int>& indices) {
  std::vector<cv::Point2f> out;
  const int n = landmarks.rows;
  for (int idx : indices) {
    if (idx < 0 || idx >= n) continue;
    float x = landmarks.at<float>(idx, 0);
    float y = landmarks.at<float>(idx, 1);
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    out.emplace_back(x, y);
  }
  return out;
}

cv::Point2f lm_center_of(const cv::Mat& landmarks, const std::vector<int>& indices) {
  auto p = lm_pts(landmarks, indices);
  if (p.empty()) return {0.0f, 0.0f};
  // numpy: float32 数组按行求 mean（结果 float32）。
  double sx = 0.0, sy = 0.0;
  for (const auto& q : p) {
    sx += q.x;
    sy += q.y;
  }
  return {static_cast<float>(sx / p.size()), static_cast<float>(sy / p.size())};
}

double estimate_head_yaw_ratio(const cv::Mat& landmarks) {
  cv::Point2f le = lm_center_of(landmarks, LEFT_EYE);
  cv::Point2f re = lm_center_of(landmarks, RIGHT_EYE);
  cv::Point2f nose = lm_center_of(landmarks, {4});
  cv::Point2f vec = re - le;
  double io = std::sqrt(static_cast<double>(vec.x) * vec.x + static_cast<double>(vec.y) * vec.y);
  if (io < 1.0) return 0.0;
  double ax = vec.x / io, ay = vec.y / io;
  cv::Point2f eye_mid((le.x + re.x) * 0.5f, (le.y + re.y) * 0.5f);
  return ((nose.x - eye_mid.x) * ax + (nose.y - eye_mid.y) * ay) / io;
}

std::vector<cv::Point2f> filter_side_brow_pts(const std::vector<cv::Point2f>& brow_pts,
                                              const std::vector<cv::Point2f>& eye_pts,
                                              double io) {
  if (brow_pts.size() < 2 || eye_pts.size() < 2) return brow_pts;
  float ex0 = eye_pts[0].x, ex1 = eye_pts[0].x, ey_top = eye_pts[0].y;
  for (const auto& p : eye_pts) {
    ex0 = std::min(ex0, p.x);
    ex1 = std::max(ex1, p.x);
    ey_top = std::min(ey_top, p.y);
  }
  std::vector<cv::Point2f> filtered;
  for (const auto& p : brow_pts) {
    if (p.x >= ex0 - io * 0.14 && p.x <= ex1 + io * 0.14 && p.y <= ey_top + io * 0.06)
      filtered.push_back(p);
  }
  if (filtered.size() >= 2) {
    float fx0 = filtered[0].x, fx1 = filtered[0].x;
    for (const auto& p : filtered) {
      fx0 = std::min(fx0, p.x);
      fx1 = std::max(fx1, p.x);
    }
    if (static_cast<double>(fx1 - fx0) >= io * 0.05) return filtered;
  }
  return brow_pts;
}

FaceGeometry get_geometry(const cv::Mat& landmarks, const cv::Size& image_size) {
  FaceGeometry g;
  cv::Point2f le = lm_center_of(landmarks, LEFT_EYE);
  cv::Point2f re = lm_center_of(landmarks, RIGHT_EYE);
  cv::Point2f eye_mid((le.x + re.x) / 2.0f, (le.y + re.y) / 2.0f);
  cv::Point2f nose_tip = lm_center_of(landmarks, {4});
  cv::Point2f mouth_left = lm_center_of(landmarks, {61});
  cv::Point2f mouth_right = lm_center_of(landmarks, {291});
  cv::Point2f chin = lm_center_of(landmarks, {152});
  cv::Point2f vec = re - le;
  double interocular = std::sqrt(static_cast<double>(vec.x) * vec.x + static_cast<double>(vec.y) * vec.y);
  if (interocular < 1) {
    interocular = static_cast<double>(std::max(image_size.width, image_size.height)) * 0.1;
    vec = {1.0f, 0.0f};
  }
  double norm = std::sqrt(static_cast<double>(vec.x) * vec.x + static_cast<double>(vec.y) * vec.y) + 1e-6;
  cv::Vec2f x_axis(static_cast<float>(vec.x / norm), static_cast<float>(vec.y / norm));
  cv::Vec2f y_axis(-x_axis[1], x_axis[0]);
  if ((chin.x - eye_mid.x) * y_axis[0] + (chin.y - eye_mid.y) * y_axis[1] < 0) y_axis = -y_axis;
  double angle = std::atan2(static_cast<double>(x_axis[1]), static_cast<double>(x_axis[0])) * 180.0 / CV_PI;
  g.left_eye_center = le;
  g.right_eye_center = re;
  g.eye_mid = eye_mid;
  g.nose_tip = nose_tip;
  g.mouth_left = mouth_left;
  g.mouth_right = mouth_right;
  g.chin = chin;
  g.interocular = interocular;
  g.face_bbox = bbox_from_landmarks(landmarks, image_size);
  g.x_axis = x_axis;
  g.y_axis = y_axis;
  g.angle_deg = angle;
  return g;
}

cv::Rect bbox_from_landmarks(const cv::Mat& landmarks, const cv::Size& image_size) {
  const int h = image_size.height, w = image_size.width;
  auto p = lm_pts(landmarks, FACE_OVAL);
  if (p.empty()) {
    for (int i = 0; i < landmarks.rows; ++i)
      p.emplace_back(landmarks.at<float>(i, 0), landmarks.at<float>(i, 1));
  }
  float minx = p[0].x, miny = p[0].y, maxx = p[0].x, maxy = p[0].y;
  for (const auto& q : p) {
    minx = std::min(minx, q.x);
    miny = std::min(miny, q.y);
    maxx = std::max(maxx, q.x);
    maxy = std::max(maxy, q.y);
  }
  int x0 = static_cast<int>(std::floor(minx));
  int y0 = static_cast<int>(std::floor(miny));
  int x1 = static_cast<int>(std::ceil(maxx));
  int y1 = static_cast<int>(std::ceil(maxy));
  x0 = std::clamp(x0, 0, w - 1);
  y0 = std::clamp(y0, 0, h - 1);
  x1 = std::clamp(x1, 0, w - 1);
  y1 = std::clamp(y1, 0, h - 1);
  return {x0, y0, std::max(0, x1 - x0 + 1), std::max(0, y1 - y0 + 1)};
}

void project_points(const std::vector<cv::Point2f>& points, const cv::Point2f& origin,
                    const cv::Vec2f& x_axis, const cv::Vec2f& y_axis,
                    std::vector<float>* u, std::vector<float>* v) {
  u->clear();
  v->clear();
  for (const auto& p : points) {
    float dx = p.x - origin.x;
    float dy = p.y - origin.y;
    u->push_back(dx * x_axis[0] + dy * x_axis[1]);
    v->push_back(dx * y_axis[0] + dy * y_axis[1]);
  }
}

}  // namespace facehi
