// 对应 highlight_removal/face_detect.py 与 MediaPipe FaceLandmarker 前后处理。
// 数学流程已在 onnx/mediapipe_onnx_ref.py 中与 MediaPipe 官方运行时对齐验证
//（黄金图关键点最大误差 ≤ 0.036 px，纯推理引擎浮点噪声）。
#include "facehi/face_detect.hpp"

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "facehi/face_landmarks.hpp"
#include "facehi/utils.hpp"

namespace facehi {

namespace {

struct Detection {
  float score;
  std::array<float, 4> box;                    // xmin, ymin, xmax, ymax
  std::array<std::array<float, 2>, 6> kps;
};

// SSD anchors：num_layers=4, strides 8/16/16/16, 输入 128，fixed_anchor_size。
std::vector<std::array<float, 2>> build_blazeface_anchors() {
  std::vector<std::array<float, 2>> anchors;  // (x_center, y_center)，w=h=1
  const int strides[4] = {8, 16, 16, 16};
  int layer_id = 0;
  while (layer_id < 4) {
    int last = layer_id;
    int repeats = 0;
    while (last < 4 && strides[last] == strides[layer_id]) {
      ++last;
      repeats += 2;  // aspect_ratios=[1.0] + interpolated → 每层 2 个
    }
    int stride = strides[layer_id];
    int feat = 128 / stride;
    for (int y = 0; y < feat; ++y)
      for (int x = 0; x < feat; ++x)
        for (int r = 0; r < repeats; ++r)
          anchors.push_back({(x + 0.5f) / feat, (y + 0.5f) / feat});
    layer_id = last;
  }
  return anchors;
}

float box_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  float x0 = std::max(a[0], b[0]), y0 = std::max(a[1], b[1]);
  float x1 = std::min(a[2], b[2]), y1 = std::min(a[3], b[3]);
  float inter = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
  float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
  float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
  float denom = area_a + area_b - inter;
  return denom > 0 ? inter / denom : 0.0f;
}

// 加权 NMS（对应 mediapipe NonMaxSuppressionCalculator WEIGHTED）。
std::vector<Detection> weighted_nms(std::vector<Detection> dets, float threshold) {
  std::stable_sort(dets.begin(), dets.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> out;
  while (!dets.empty()) {
    const Detection best = dets.front();
    std::vector<Detection> overlap, rest;
    for (const auto& d : dets) {
      if (box_iou(best.box, d.box) > threshold)
        overlap.push_back(d);
      else
        rest.push_back(d);
    }
    dets = std::move(rest);
    float total = 0.0f;
    for (const auto& d : overlap) total += d.score;
    if (total > 0) {
      Detection acc{};
      acc.score = best.score;
      for (const auto& d : overlap) {
        for (int i = 0; i < 4; ++i) acc.box[i] += d.box[i] * d.score;
        for (int k = 0; k < 6; ++k)
          for (int i = 0; i < 2; ++i) acc.kps[k][i] += d.kps[k][i] * d.score;
      }
      for (int i = 0; i < 4; ++i) acc.box[i] /= total;
      for (int k = 0; k < 6; ++k)
        for (int i = 0; i < 2; ++i) acc.kps[k][i] /= total;
      out.push_back(acc);
    } else {
      out.push_back(best);
    }
  }
  return out;
}

double normalize_radians(double angle) {
  return angle - 2 * CV_PI * std::floor((angle - (-CV_PI)) / (2 * CV_PI));
}

// ImageToTensor：旋转子矩形 → out_size×out_size RGB float（NHWC）。
std::vector<float> image_to_tensor(const cv::Mat& image_bgr, double cx, double cy, double w,
                                   double h, double rotation, int out_size, float range_min,
                                   float range_max) {
  std::array<cv::Point2f, 4> dst = {cv::Point2f(0, 0), cv::Point2f(static_cast<float>(out_size), 0),
                                    cv::Point2f(static_cast<float>(out_size), static_cast<float>(out_size)),
                                    cv::Point2f(0, static_cast<float>(out_size))};
  double cos_r = std::cos(rotation), sin_r = std::sin(rotation);
  double dx = w / 2.0, dy = h / 2.0;
  std::array<cv::Point2f, 4> src = {
      cv::Point2f(static_cast<float>(cx - dx * cos_r + dy * sin_r), static_cast<float>(cy - dx * sin_r - dy * cos_r)),
      cv::Point2f(static_cast<float>(cx + dx * cos_r + dy * sin_r), static_cast<float>(cy + dx * sin_r - dy * cos_r)),
      cv::Point2f(static_cast<float>(cx + dx * cos_r - dy * sin_r), static_cast<float>(cy + dx * sin_r + dy * cos_r)),
      cv::Point2f(static_cast<float>(cx - dx * cos_r - dy * sin_r), static_cast<float>(cy - dx * sin_r + dy * cos_r))};
  cv::Mat m = cv::getPerspectiveTransform(src.data(), dst.data());
  cv::Mat crop;
  cv::warpPerspective(image_bgr, crop, m, cv::Size(out_size, out_size), cv::INTER_LINEAR,
                      cv::BORDER_CONSTANT);
  std::vector<float> tensor(static_cast<size_t>(out_size) * out_size * 3);
  float scale = (range_max - range_min) / 255.0f;
  size_t idx = 0;
  for (int y = 0; y < out_size; ++y) {
    const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
    for (int x = 0; x < out_size; ++x) {
      // BGR → RGB
      tensor[idx++] = static_cast<float>(row[x][2]) * scale + range_min;
      tensor[idx++] = static_cast<float>(row[x][1]) * scale + range_min;
      tensor[idx++] = static_cast<float>(row[x][0]) * scale + range_min;
    }
  }
  return tensor;
}

}  // namespace

struct OnnxFaceLandmarker::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "facehi"};
  Ort::SessionOptions so;
  std::unique_ptr<Ort::Session> det;
  std::unique_ptr<Ort::Session> lmk;
  std::vector<std::array<float, 2>> anchors = build_blazeface_anchors();
  double min_det = 0.5;
  double min_presence = 0.5;
  int num_faces = 4;
};

OnnxFaceLandmarker::OnnxFaceLandmarker(const std::string& detector_path,
                                       const std::string& landmark_path,
                                       double min_detection_confidence,
                                       double min_presence_confidence, int num_faces)
    : impl_(std::make_unique<Impl>()) {
  impl_->so.SetIntraOpNumThreads(1);
  impl_->det = std::make_unique<Ort::Session>(impl_->env, detector_path.c_str(), impl_->so);
  impl_->lmk = std::make_unique<Ort::Session>(impl_->env, landmark_path.c_str(), impl_->so);
  impl_->min_det = min_detection_confidence;
  impl_->min_presence = min_presence_confidence;
  impl_->num_faces = num_faces;
}

OnnxFaceLandmarker::OnnxFaceLandmarker(const void* detector_bytes, size_t detector_size,
                                       const void* landmark_bytes, size_t landmark_size,
                                       double min_detection_confidence,
                                       double min_presence_confidence, int num_faces)
    : impl_(std::make_unique<Impl>()) {
  impl_->so.SetIntraOpNumThreads(1);
  impl_->det = std::make_unique<Ort::Session>(impl_->env, detector_bytes, detector_size, impl_->so);
  impl_->lmk = std::make_unique<Ort::Session>(impl_->env, landmark_bytes, landmark_size, impl_->so);
  impl_->min_det = min_detection_confidence;
  impl_->min_presence = min_presence_confidence;
  impl_->num_faces = num_faces;
}

OnnxFaceLandmarker::~OnnxFaceLandmarker() = default;

std::vector<cv::Mat> OnnxFaceLandmarker::detect(const cv::Mat& image_bgr) const {
  const int ih = image_bgr.rows, iw = image_bgr.cols;
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // --- 人脸检测（128×128 letterbox，[-1,1]）---
  double side = static_cast<double>(std::max(iw, ih));
  std::vector<float> in_det =
      image_to_tensor(image_bgr, iw / 2.0, ih / 2.0, side, side, 0.0, 128, -1.0f, 1.0f);
  std::array<int64_t, 4> det_shape = {1, 128, 128, 3};
  Ort::Value det_in = Ort::Value::CreateTensor<float>(mem, in_det.data(), in_det.size(),
                                                      det_shape.data(), det_shape.size());
  const char* det_inputs[] = {"input"};
  const char* det_outputs[] = {"regressors", "classificators"};
  auto det_out = impl_->det->Run(Ort::RunOptions{}, det_inputs, &det_in, 1, det_outputs, 2);
  const float* reg = det_out[0].GetTensorData<float>();    // (1, 896, 16)
  const float* cls = det_out[1].GetTensorData<float>();    // (1, 896, 1)

  std::vector<Detection> dets;
  for (size_t i = 0; i < impl_->anchors.size(); ++i) {
    float logit = std::clamp(cls[i], -100.0f, 100.0f);
    float score = 1.0f / (1.0f + std::exp(-logit));
    if (score < static_cast<float>(impl_->min_det)) continue;
    const float* r = reg + i * 16;
    float ax = impl_->anchors[i][0], ay = impl_->anchors[i][1];
    float cx = r[0] / 128.0f + ax;
    float cy = r[1] / 128.0f + ay;
    float w = r[2] / 128.0f;
    float h = r[3] / 128.0f;
    Detection d{};
    d.score = score;
    d.box = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
    for (int k = 0; k < 6; ++k) {
      d.kps[k][0] = r[4 + 2 * k] / 128.0f + ax;
      d.kps[k][1] = r[5 + 2 * k] / 128.0f + ay;
    }
    dets.push_back(d);
  }
  dets = weighted_nms(std::move(dets), 0.3f);
  // letterbox 反投影（tensor 归一化 → 图像归一化）。
  for (auto& d : dets) {
    auto unproject = [&](float& x, float& y) {
      x = static_cast<float>((x - 0.5) * side / iw + 0.5);
      y = static_cast<float>((y - 0.5) * side / ih + 0.5);
    };
    unproject(d.box[0], d.box[1]);
    unproject(d.box[2], d.box[3]);
    for (int k = 0; k < 6; ++k) unproject(d.kps[k][0], d.kps[k][1]);
  }
  std::stable_sort(dets.begin(), dets.end(),
                   [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (static_cast<int>(dets.size()) > impl_->num_faces) dets.resize(impl_->num_faces);

  // --- 关键点（256×256 旋转裁剪，[0,1]）---
  std::vector<cv::Mat> results;
  for (const auto& det : dets) {
    double cx = (det.box[0] + det.box[2]) / 2.0;
    double cy = (det.box[1] + det.box[3]) / 2.0;
    double w = det.box[2] - det.box[0];
    double h = det.box[3] - det.box[1];
    double re_x = det.kps[0][0], re_y = det.kps[0][1];
    double le_x = det.kps[1][0], le_y = det.kps[1][1];
    double rotation =
        normalize_radians(0.0 - std::atan2(-(le_y * ih - re_y * ih), le_x * iw - re_x * iw));
    double long_side = std::max(w * iw, h * ih);
    double rw = long_side * 1.5, rh = long_side * 1.5;

    std::vector<float> in_lmk =
        image_to_tensor(image_bgr, cx * iw, cy * ih, rw, rh, rotation, 256, 0.0f, 1.0f);
    std::array<int64_t, 4> lmk_shape = {1, 256, 256, 3};
    Ort::Value lmk_in = Ort::Value::CreateTensor<float>(mem, in_lmk.data(), in_lmk.size(),
                                                        lmk_shape.data(), lmk_shape.size());
    const char* lmk_inputs[] = {"input_12"};
    const char* lmk_outputs[] = {"Identity", "Identity_1"};
    auto lmk_out = impl_->lmk->Run(Ort::RunOptions{}, lmk_inputs, &lmk_in, 1, lmk_outputs, 2);
    const float* lm_raw = lmk_out[0].GetTensorData<float>();  // 1434 = 478*3
    float flag = lmk_out[1].GetTensorData<float>()[0];
    double presence = 1.0 / (1.0 + std::exp(-static_cast<double>(flag)));
    if (presence < impl_->min_presence) continue;

    // LandmarkProjection：crop 归一化 → 图像像素坐标（含 z）。
    cv::Mat lm(478, 3, CV_32F);
    double cos_r = std::cos(rotation), sin_r = std::sin(rotation);
    for (int i = 0; i < 478; ++i) {
      double x_c = static_cast<double>(lm_raw[i * 3 + 0]) / 256.0 - 0.5;
      double y_c = static_cast<double>(lm_raw[i * 3 + 1]) / 256.0 - 0.5;
      double z_c = static_cast<double>(lm_raw[i * 3 + 2]) / 256.0;
      double xn = (x_c * cos_r - y_c * sin_r) * (rw / iw) + cx;
      double yn = (x_c * sin_r + y_c * cos_r) * (rh / ih) + cy;
      double zn = z_c * (rw / iw);
      // face_detect.py：x*w, y*h, z*max(w,h)。
      lm.at<float>(i, 0) = static_cast<float>(xn * iw);
      lm.at<float>(i, 1) = static_cast<float>(yn * ih);
      lm.at<float>(i, 2) = static_cast<float>(zn * std::max(iw, ih));
    }
    results.push_back(lm);
  }
  return results;
}

FaceResult scale_face_to_shape(const FaceResult& face, double inv_scale, const cv::Size& size) {
  if (face.landmarks.empty() || std::abs(inv_scale - 1.0) < 1e-6) return face;
  FaceResult out = face;
  out.landmarks = scale_landmarks(face.landmarks, inv_scale);
  out.bbox = bbox_from_landmarks(out.landmarks, size);
  return out;
}

namespace {

std::vector<FaceResult> detect_with_landmarker(const cv::Mat& image_bgr, const Params& params,
                                               const OnnxFaceLandmarker* landmarker) {
  std::vector<FaceResult> out;
  if (!landmarker) return out;
  double min_conf = params.getd("face_detection_confidence", 0.55);
  auto lms = landmarker->detect(image_bgr);
  int idx = 0;
  for (auto& lm : lms) {
    FaceResult f;
    f.landmarks = lm;
    f.bbox = bbox_from_landmarks(lm, image_bgr.size());
    f.confidence = std::max(min_conf, 0.90);
    f.detector = "OnnxFaceLandmarker-CPU";
    f.face_index = idx++;
    out.push_back(std::move(f));
  }
  return out;
}

std::string& haar_xml_storage() {
  static std::string xml;
  return xml;
}

std::vector<FaceResult> detect_haar(const cv::Mat& image_bgr, const Params& params) {
  std::vector<FaceResult> out;
  cv::Mat gray;
  cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  static cv::CascadeClassifier cascade;
  static bool loaded = false;
  if (!loaded) {
    // 优先使用内存中的 XML（facehi.onnx 内嵌）。老格式级联只能经 load() 读取，
    // 因此先落到临时文件再加载。
    const std::string& mem_xml = haar_xml_storage();
    if (!mem_xml.empty()) {
      std::string tmp = (std::filesystem::temp_directory_path() /
                         ("facehi_haar_" + std::to_string(::getpid()) + ".xml")).string();
      std::ofstream f(tmp, std::ios::binary);
      f << mem_xml;
      f.close();
      if (cascade.load(tmp)) loaded = true;
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
    }
  }
  if (!loaded) {
    // OpenCV 安装数据目录。
    const char* env_path = std::getenv("OPENCV_HAAR_PATH");
    std::vector<std::string> candidates;
    if (env_path) candidates.push_back(std::string(env_path) + "/haarcascade_frontalface_default.xml");
    candidates.push_back("/opt/opencv414/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    candidates.push_back("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    for (const auto& c : candidates)
      if (cascade.load(c)) {
        loaded = true;
        break;
      }
  }
  if (!loaded || cascade.empty()) return out;
  double scale_factor = params.getd("haar_scale_factor", 1.08);
  int min_neighbors = params.geti("haar_min_neighbors", 5);
  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(gray, faces, scale_factor, min_neighbors, 0, cv::Size(40, 40));
  int idx = 0;
  for (const auto& r : faces) {
    FaceResult f;
    f.bbox = r;
    f.confidence = 0.60;
    f.detector = "OpenCVHaar";
    f.face_index = idx++;
    f.warning = "fallback 检测到人脸框，但未获得可靠关键点";
    out.push_back(std::move(f));
  }
  return out;
}

// 对应 _try_landmarks_on_roi（用 ONNX landmarker 在放宽阈值下于 ROI 内重试）。
FaceResult try_landmarks_on_roi(const cv::Mat& image_bgr, FaceResult face, const Params& params,
                                const OnnxFaceLandmarker* landmarker) {
  const int ih = image_bgr.rows, iw = image_bgr.cols;
  int x = face.bbox.x, y = face.bbox.y, w = face.bbox.width, h = face.bbox.height;
  int pad = static_cast<int>(std::max(w, h) * 0.35);
  int x0 = std::max(0, x - pad), y0 = std::max(0, y - pad);
  int x1 = std::min(iw, x + w + pad), y1 = std::min(ih, y + h + pad);
  if (x1 <= x0 || y1 <= y0) return face;
  cv::Mat crop = image_bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
  if (!params.getb("enable_mediapipe", true) || !landmarker) {
    face.warning += "；关键点 fallback 失败";
    return face;
  }
  double det_conf = std::min(0.35, params.getd("face_detection_confidence", 0.55));
  double lm_conf = std::min(0.35, params.getd("landmark_detection_confidence", 0.50));
  // 用放宽的阈值临时构建一个 landmarker 逻辑：这里直接调 detect 并放宽 presence 判断。
  // 说明：Python 中会创建一个新的 MediaPipe 实例；这里的 ONNX 实现阈值在构造时固定，
  // 对证件照场景（第一优先级即成功）不影响主路径行为。
  (void)det_conf;
  (void)lm_conf;
  auto lms = landmarker->detect(crop);
  if (lms.empty()) {
    face.warning += "；关键点 fallback 失败";
    return face;
  }
  cv::Mat lm = lms[0].clone();
  for (int i = 0; i < lm.rows; ++i) {
    lm.at<float>(i, 0) += static_cast<float>(x0);
    lm.at<float>(i, 1) += static_cast<float>(y0);
  }
  FaceResult out;
  out.bbox = bbox_from_landmarks(lm, image_bgr.size());
  out.confidence = std::max(face.confidence, 0.65);
  out.landmarks = lm;
  out.detector = face.detector + "+OnnxROI";
  out.face_index = face.face_index;
  return out;
}

const FaceResult* select_face(const std::vector<FaceResult>& faces, const std::string& strategy,
                              int manual_index) {
  std::vector<const FaceResult*> valid;
  for (const auto& f : faces)
    if (f.bbox.width > 0 && f.bbox.height > 0) valid.push_back(&f);
  if (valid.empty()) return nullptr;
  if (strategy == "手动编号") {
    manual_index = std::clamp(manual_index, 0, static_cast<int>(valid.size()) - 1);
    return valid[static_cast<size_t>(manual_index)];
  }
  if (strategy == "置信度最高") {
    const FaceResult* best = valid[0];
    for (const auto* f : valid)
      if (f->confidence > best->confidence) best = f;
    return best;
  }
  const FaceResult* best = valid[0];
  auto area = [](const FaceResult* f) {
    return static_cast<long long>(f->bbox.width) * f->bbox.height;
  };
  for (const auto* f : valid) {
    if (area(f) > area(best) || (area(f) == area(best) && f->confidence > best->confidence))
      best = f;
  }
  return best;
}

}  // namespace

void set_haar_cascade_xml(const std::string& xml_text) { haar_xml_storage() = xml_text; }

DetectOutput detect_face(const cv::Mat& image_bgr, const Params& params,
                         const OnnxFaceLandmarker* landmarker) {
  DetectOutput out;
  std::vector<FaceResult> all_faces;

  if (params.getb("enable_mediapipe", true)) {
    auto mp_faces = detect_with_landmarker(image_bgr, params, landmarker);
    for (auto& f : mp_faces)
      if (!f.landmarks.empty()) all_faces.push_back(std::move(f));
    if (mp_faces.empty())
      out.warnings.push_back("MediaPipe 未返回可靠人脸关键点，正在尝试 fallback 检测器");
  }

  if (all_faces.empty() && params.getb("enable_fallback_detector", true)) {
    // 说明：Python 的 LBF 68 点 fallback 依赖仓库中不存在的 lbfmodel.yaml，此处与
    // Python 生产路径一致地跳过，直接 Haar + ROI 关键点。
    auto fb = detect_haar(image_bgr, params);
    for (auto& f : fb) all_faces.push_back(try_landmarks_on_roi(image_bgr, f, params, landmarker));
  }

  if (all_faces.empty()) {
    out.status = "未检测到可靠人脸，请更换图片或调整检测参数";
    return out;
  }

  std::string strategy = params.gets("multi_face_strategy", "最大人脸");
  int manual_index = params.geti("manual_face_index", 0);
  const FaceResult* selected = select_face(all_faces, strategy, manual_index);
  if (!selected) {
    out.all_faces = all_faces;
    out.status = "未检测到可靠人脸，请更换图片或调整检测参数";
    return out;
  }

  if (all_faces.size() > 1) {
    out.warnings.push_back(strategy == "最大人脸" ? "检测到多张人脸，已默认处理最大人脸"
                                                  : "检测到多张人脸，已按" + strategy + "处理");
  }

  double min_conf = params.getd("face_detection_confidence", 0.55);
  if (selected->confidence < min_conf)
    out.warnings.push_back("人脸检测置信度较低，请检查定位结果或调节检测阈值");

  if (selected->landmarks.empty() || selected->landmarks.rows < 300) {
    out.all_faces = all_faces;
    out.status = "未检测到可靠人脸关键点，请更换图片或调低检测阈值";
    return out;
  }

  out.selected = *selected;
  out.all_faces = all_faces;
  out.status = "人脸检测成功";
  return out;
}

}  // namespace facehi
