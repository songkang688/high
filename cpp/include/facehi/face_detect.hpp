// 对应 highlight_removal/face_detect.py：ONNX Runtime 版 MediaPipe FaceLandmarker。
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facehi/common.hpp"

namespace facehi {

// 对应 FaceResult dataclass。
struct FaceResult {
  cv::Rect bbox;
  double confidence = 0.0;
  cv::Mat landmarks;  // 478 x 3 CV_32F（原图像素坐标；z 已乘 max(w,h)）
  std::string detector;
  int face_index = 0;
  std::string warning;
};

// 对应 DetectOutput dataclass。
struct DetectOutput {
  std::optional<FaceResult> selected;
  std::vector<FaceResult> all_faces;
  std::string status;
  std::vector<std::string> warnings;
};

// ONNX 版 MediaPipe FaceLandmarker（BlazeFace 检测 + 478 点关键点 + 官方前后处理）。
class OnnxFaceLandmarker {
 public:
  // detector_path / landmark_path：从 face_landmarker.task 拆出并转换的 ONNX 模型。
  OnnxFaceLandmarker(const std::string& detector_path, const std::string& landmark_path,
                     double min_detection_confidence, double min_presence_confidence,
                     int num_faces);
  ~OnnxFaceLandmarker();

  // 返回若干人脸的 478x3 关键点（图像像素坐标；z 乘 max(w,h)）。
  std::vector<cv::Mat> detect(const cv::Mat& image_bgr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// 对应 face_detect.detect_face（MediaPipe 优先，Haar+ROI 兜底）。
DetectOutput detect_face(const cv::Mat& image_bgr, const Params& params,
                         const OnnxFaceLandmarker* landmarker);

// 对应 scale_face_to_shape。
FaceResult scale_face_to_shape(const FaceResult& face, double inv_scale, const cv::Size& size);

}  // namespace facehi
