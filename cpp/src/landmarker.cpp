#include "high_removal/landmarker.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace hr {

namespace {

constexpr int kDetectorInput = 128;
constexpr int kLandmarkInput = 256;
constexpr int kNumLandmarks = 478;
constexpr double kMinSuppressionThreshold = 0.3;
constexpr double kRectScale = 1.5;

struct NormRect {
    double xc, yc, w, h, rotation;  // 归一化坐标 + 弧度
};

struct Detection {
    double x0, y0, w, h;      // 归一化 bbox
    double kp[6][2];          // 6 个关键点
    double score;
};

// ssd_anchors_calculator.cc：BlazeFace 短距 896 anchors（fixed_anchor_size，只需中心）。
std::vector<cv::Point2d> buildAnchors() {
    const int strides[] = {8, 16, 16, 16};
    std::vector<cv::Point2d> anchors;
    int layerId = 0;
    while (layerId < 4) {
        int last = layerId;
        int repeats = 0;
        while (last < 4 && strides[last] == strides[layerId]) {
            ++last;
            repeats += 2;  // aspect_ratio 1.0 + interpolated 1.0
        }
        const int stride = strides[layerId];
        const int cells = kDetectorInput / stride;
        for (int y = 0; y < cells; ++y) {
            const double yc = (y + 0.5) * stride / static_cast<double>(kDetectorInput);
            for (int x = 0; x < cells; ++x) {
                const double xc = (x + 0.5) * stride / static_cast<double>(kDetectorInput);
                for (int r = 0; r < repeats; ++r) anchors.emplace_back(xc, yc);
            }
        }
        layerId = last;
    }
    return anchors;
}

// image_to_tensor_converter_opencv.cc：旋转矩形 → warpPerspective → RGB 归一化。
cv::Mat imageToTensor(const cv::Mat& imageBgr, const NormRect& roi, int outSize, float outMin, float outMax) {
    const int w = imageBgr.cols, h = imageBgr.rows;
    const cv::RotatedRect rotated(cv::Point2f(static_cast<float>(roi.xc * w), static_cast<float>(roi.yc * h)),
                                  cv::Size2f(static_cast<float>(roi.w * w), static_cast<float>(roi.h * h)),
                                  static_cast<float>(roi.rotation * 180.0 / CV_PI));
    cv::Point2f srcPts[4];
    rotated.points(srcPts);
    const auto fs = static_cast<float>(outSize);
    const cv::Point2f dstPts[4] = {{0.f, fs}, {0.f, 0.f}, {fs, 0.f}, {fs, fs}};
    const cv::Mat matrix = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat warped;
    cv::warpPerspective(imageBgr, warped, matrix, cv::Size(outSize, outSize), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::Mat rgb;
    cv::cvtColor(warped, rgb, cv::COLOR_BGR2RGB);
    cv::Mat tensor;
    rgb.convertTo(tensor, CV_32FC3, (outMax - outMin) / 255.0, outMin);
    return tensor;  // HWC RGB float32
}

double iou(const Detection& a, const Detection& b) {
    const double ix0 = std::max(a.x0, b.x0);
    const double iy0 = std::max(a.y0, b.y0);
    const double ix1 = std::min(a.x0 + a.w, b.x0 + b.w);
    const double iy1 = std::min(a.y0 + a.h, b.y0 + b.h);
    const double iw = std::max(0.0, ix1 - ix0);
    const double ih = std::max(0.0, iy1 - iy0);
    const double inter = iw * ih;
    const double denom = a.w * a.h + b.w * b.h - inter;
    return denom > 0 ? inter / denom : 0.0;
}

// non_max_suppression_calculator.cc 的 WEIGHTED 模式。
std::vector<Detection> weightedNms(std::vector<Detection> dets) {
    std::stable_sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> out;
    while (!dets.empty()) {
        const Detection top = dets.front();
        std::vector<Detection> candidates, remained;
        for (const auto& d : dets) {
            if (iou(d, top) > kMinSuppressionThreshold) {
                candidates.push_back(d);
            } else {
                remained.push_back(d);
            }
        }
        Detection weighted = top;
        if (!candidates.empty()) {
            double total = 0, wx0 = 0, wy0 = 0, wx1 = 0, wy1 = 0;
            double wkp[6][2] = {};
            for (const auto& d : candidates) {
                total += d.score;
                wx0 += d.x0 * d.score;
                wy0 += d.y0 * d.score;
                wx1 += (d.x0 + d.w) * d.score;
                wy1 += (d.y0 + d.h) * d.score;
                for (int k = 0; k < 6; ++k) {
                    wkp[k][0] += d.kp[k][0] * d.score;
                    wkp[k][1] += d.kp[k][1] * d.score;
                }
            }
            weighted.x0 = wx0 / total;
            weighted.y0 = wy0 / total;
            weighted.w = wx1 / total - weighted.x0;
            weighted.h = wy1 / total - weighted.y0;
            for (int k = 0; k < 6; ++k) {
                weighted.kp[k][0] = wkp[k][0] / total;
                weighted.kp[k][1] = wkp[k][1] / total;
            }
        }
        out.push_back(weighted);
        dets = std::move(remained);
    }
    return out;
}

double normalizeRadians(double angle) {
    return angle - 2.0 * CV_PI * std::floor((angle + CV_PI) / (2.0 * CV_PI));
}

}  // namespace

struct FaceLandmarkerOrt::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "high_removal"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> det;
    std::unique_ptr<Ort::Session> lmk;
    std::string detInputName, lmkInputName;
    std::vector<std::string> detOutputNames, lmkOutputNames;
    std::vector<cv::Point2d> anchors = buildAnchors();
    int numFaces = 4;
    float minDetConf = 0.5f;
    float minPresence = 0.5f;

    void initThresholds(int faces, float minDet, float minPres) {
        numFaces = faces;
        minDetConf = minDet;
        minPresence = minPres;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    void cacheIoNames() {
        Ort::AllocatorWithDefaultOptions alloc;
        detInputName = det->GetInputNameAllocated(0, alloc).get();
        lmkInputName = lmk->GetInputNameAllocated(0, alloc).get();
        for (size_t i = 0; i < det->GetOutputCount(); ++i) {
            detOutputNames.push_back(det->GetOutputNameAllocated(i, alloc).get());
        }
        for (size_t i = 0; i < lmk->GetOutputCount(); ++i) {
            lmkOutputNames.push_back(lmk->GetOutputNameAllocated(i, alloc).get());
        }
    }

    static std::vector<Ort::Value> run(Ort::Session& session, const std::string& inputName,
                                       const std::vector<std::string>& outputNames, const cv::Mat& tensorHwc) {
        const std::array<int64_t, 4> shape{1, tensorHwc.rows, tensorHwc.cols, 3};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(tensorHwc.ptr<float>()), static_cast<size_t>(tensorHwc.total() * 3),
            shape.data(), shape.size());
        const char* inNames[] = {inputName.c_str()};
        std::vector<const char*> outNames;
        for (const auto& n : outputNames) outNames.push_back(n.c_str());
        return session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames.data(), outNames.size());
    }
};

FaceLandmarkerOrt::FaceLandmarkerOrt(const std::string& modelDir, int numFaces,
                                     float minDetectionConfidence, float minPresenceConfidence)
    : impl_(std::make_unique<Impl>()) {
    impl_->initThresholds(numFaces, minDetectionConfidence, minPresenceConfidence);
    impl_->det = std::make_unique<Ort::Session>(impl_->env, (modelDir + "/face_detector.onnx").c_str(), impl_->opts);
    impl_->lmk = std::make_unique<Ort::Session>(impl_->env, (modelDir + "/face_landmarks_detector.onnx").c_str(), impl_->opts);
    impl_->cacheIoNames();
}

FaceLandmarkerOrt::FaceLandmarkerOrt(const void* detectorData, size_t detectorSize,
                                     const void* landmarksData, size_t landmarksSize,
                                     int numFaces, float minDetectionConfidence, float minPresenceConfidence)
    : impl_(std::make_unique<Impl>()) {
    impl_->initThresholds(numFaces, minDetectionConfidence, minPresenceConfidence);
    impl_->det = std::make_unique<Ort::Session>(impl_->env, detectorData, detectorSize, impl_->opts);
    impl_->lmk = std::make_unique<Ort::Session>(impl_->env, landmarksData, landmarksSize, impl_->opts);
    impl_->cacheIoNames();
}

FaceLandmarkerOrt::~FaceLandmarkerOrt() = default;

std::vector<DetectedFace> FaceLandmarkerOrt::detect(const cv::Mat& imageBgr) const {
    auto& impl = *impl_;
    const int w = imageBgr.cols, h = imageBgr.rows;

    // 1) 检测：整图 letterbox 成正方形 ROI（keep_aspect_ratio=true）。
    double padW = static_cast<double>(w), padH = static_cast<double>(h);
    if (padH > padW) {
        padW = padH;
    } else {
        padH = padW;
    }
    const NormRect padded{0.5, 0.5, padW / w, padH / h, 0.0};
    const cv::Mat detTensor = imageToTensor(imageBgr, padded, kDetectorInput, -1.f, 1.f);
    auto detOut = Impl::run(*impl.det, impl.detInputName, impl.detOutputNames, detTensor);

    // 输出按名称匹配：regressors (1,896,16)、classificators (1,896,1)。
    const float* rawBoxes = nullptr;
    const float* rawScores = nullptr;
    for (size_t i = 0; i < detOut.size(); ++i) {
        const auto info = detOut[i].GetTensorTypeAndShapeInfo();
        if (info.GetElementCount() == 896 * 16) rawBoxes = detOut[i].GetTensorData<float>();
        if (info.GetElementCount() == 896) rawScores = detOut[i].GetTensorData<float>();
    }
    CV_Assert(rawBoxes && rawScores);

    // 2) tensors_to_detections_calculator.cc 解码（张量空间），再投影回原图归一化坐标。
    const double pwAbs = padded.w * w;
    const double phAbs = padded.h * h;
    std::vector<Detection> dets;
    for (int i = 0; i < 896; ++i) {
        const double logit = std::max(-100.0, std::min(100.0, static_cast<double>(rawScores[i])));
        const double score = 1.0 / (1.0 + std::exp(-logit));
        if (score < impl.minDetConf) continue;
        const float* r = rawBoxes + i * 16;
        const double ax = impl.anchors[i].x, ay = impl.anchors[i].y;
        const double xc = r[0] / kDetectorInput + ax;
        const double yc = r[1] / kDetectorInput + ay;
        const double bw = r[2] / kDetectorInput;
        const double bh = r[3] / kDetectorInput;
        Detection d;
        d.score = score;
        // 投影：x_img = pad_cx + (xt-0.5)*pad_w，再归一化到原图。
        const double nx = (padded.xc * w + (xc - 0.5) * pwAbs) / w;
        const double ny = (padded.yc * h + (yc - 0.5) * phAbs) / h;
        const double nbw = bw * pwAbs / w;
        const double nbh = bh * phAbs / h;
        d.x0 = nx - nbw / 2;
        d.y0 = ny - nbh / 2;
        d.w = nbw;
        d.h = nbh;
        for (int k = 0; k < 6; ++k) {
            const double kx = r[4 + 2 * k] / kDetectorInput + ax;
            const double ky = r[4 + 2 * k + 1] / kDetectorInput + ay;
            d.kp[k][0] = (padded.xc * w + (kx - 0.5) * pwAbs) / w;
            d.kp[k][1] = (padded.yc * h + (ky - 0.5) * phAbs) / h;
        }
        dets.push_back(d);
    }

    auto merged = weightedNms(std::move(dets));
    if (static_cast<int>(merged.size()) > impl.numFaces) merged.resize(impl.numFaces);

    // 3) 每个检测框 → 旋转矩形 → 256x256 → 478 点。
    std::vector<DetectedFace> results;
    for (const auto& det : merged) {
        // detections_to_rects_calculator.cc：关键点 0(左眼)→1(右眼)，目标角 0。
        const double x0 = det.kp[0][0] * w, y0 = det.kp[0][1] * h;
        const double x1 = det.kp[1][0] * w, y1 = det.kp[1][1] * h;
        const double rotation = normalizeRadians(-std::atan2(-(y1 - y0), x1 - x0));
        // rect_transformation_calculator.cc：scale 1.5 + square_long。
        const double longSide = std::max(det.w * w * kRectScale, det.h * h * kRectScale);
        const NormRect rect{det.x0 + det.w / 2, det.y0 + det.h / 2, longSide / w, longSide / h, rotation};

        const cv::Mat lmkTensor = imageToTensor(imageBgr, rect, kLandmarkInput, 0.f, 1.f);
        auto lmkOut = Impl::run(*impl.lmk, impl.lmkInputName, impl.lmkOutputNames, lmkTensor);
        const float* rawLandmarks = nullptr;
        const float* rawFlag = nullptr;
        for (size_t i = 0; i < lmkOut.size(); ++i) {
            const auto info = lmkOut[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() == kNumLandmarks * 3) rawLandmarks = lmkOut[i].GetTensorData<float>();
            if (info.GetElementCount() == 1 && info.GetShape().size() == 4) rawFlag = lmkOut[i].GetTensorData<float>();
        }
        CV_Assert(rawLandmarks && rawFlag);
        const float presence = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(rawFlag[0]))));
        if (presence < impl.minPresence) continue;

        // landmark_projection_calculator.cc：张量空间 → 归一化图像坐标 → 像素坐标。
        DetectedFace face;
        face.presence = presence;
        face.landmarks.resize(kNumLandmarks);
        const double c = std::cos(rect.rotation), s = std::sin(rect.rotation);
        for (int i = 0; i < kNumLandmarks; ++i) {
            const double lx = rawLandmarks[i * 3 + 0] / kLandmarkInput - 0.5;
            const double ly = rawLandmarks[i * 3 + 1] / kLandmarkInput - 0.5;
            const double lz = rawLandmarks[i * 3 + 2] / kLandmarkInput;
            const double nx = (lx * c - ly * s) * rect.w + rect.xc;
            const double ny = (lx * s + ly * c) * rect.h + rect.yc;
            const double nz = lz * rect.w;
            face.landmarks[i] = cv::Vec3f(static_cast<float>(nx * w), static_cast<float>(ny * h),
                                          static_cast<float>(nz * std::max(w, h)));
        }
        results.push_back(std::move(face));
    }
    return results;
}

}  // namespace hr
