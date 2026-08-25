// 对应 highlight_removal/pipeline.py（enable_visualization=false 路径，与 CLI 一致）。
#include "facehi/pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include "facehi/highlight_detect.hpp"
#include "facehi/remove_highlight.hpp"
#include "facehi/utils.hpp"

namespace facehi {

namespace {

// 对应 _merge_removal_params + _removal_params。
Params removal_params_of(const Config& config) {
  Params p = config.highlight_removal;
  p.merge(config.highlight_detection);
  p.set("enable_roi_removal", config.pipeline.getb("enable_roi_removal", true));
  p.set("roi_padding_ratio", config.pipeline.getd("roi_padding_ratio", 0.12));
  return p;
}

void dedupe_warnings(std::vector<std::string>* warnings) {
  std::vector<std::string> out;
  for (const auto& w : *warnings) {
    if (w.empty()) continue;
    bool seen = false;
    for (const auto& o : out)
      if (o == w) {
        seen = true;
        break;
      }
    if (!seen) out.push_back(w);
  }
  *warnings = std::move(out);
}

struct HighlightMasks {
  cv::Mat hard;
  cv::Mat soft;
  std::vector<std::string> warnings;
  RegionMasks regions;
};

// 对应 _detect_highlight_masks（省略仅计时用的 1/4 区域生成）。
HighlightMasks detect_highlight_masks(const cv::Mat& image_bgr, const FaceResult& face,
                                      const Config& config) {
  auto [face_scale, highlight_scale, is_compromise] = resolve_process_scales(config.pipeline);
  (void)face_scale;
  (void)is_compromise;
  bool refine = config.pipeline.getb("refine_upsampled_masks", true);
  Params hd_params = config.highlight_detection;
  const Params& region_params = config.regions;

  HighlightMasks out;
  bool use_full_highlight = highlight_scale >= 0.999 || refine;
  if (use_full_highlight) {
    out.regions = build_face_regions(image_bgr, face.landmarks, region_params);
    HighlightOutput hl = detect_highlight(image_bgr, out.regions, hd_params);
    out.hard = hl.hard_mask;
    out.soft = hl.soft_mask;
    out.warnings = hl.warnings;
    return out;
  }

  double hl_scale = highlight_scale;
  cv::Mat small = resize_for_process(image_bgr, hl_scale);
  cv::Mat lm_hl = scale_landmarks(face.landmarks, hl_scale);
  RegionMasks regions_hl = build_face_regions(small, lm_hl, region_params);
  Params hd_scaled = scaled_highlight_params(hd_params, hl_scale);
  HighlightOutput hl = detect_highlight(small, regions_hl, hd_scaled);

  out.hard = upsample_mask_hard(hl.hard_mask, image_bgr.size());
  out.soft = upsample_mask_soft(hl.soft_mask, image_bgr.size());
  out.regions = build_face_regions(image_bgr, face.landmarks, region_params);
  out.warnings = hl.warnings;
  return out;
}

PipelineOutput process_core(const cv::Mat& image_bgr, const FaceResult& face, int detection_count,
                            const Config& config) {
  PipelineOutput out;
  HighlightMasks hm = detect_highlight_masks(image_bgr, face, config);

  RemovalOutput removal =
      remove_highlight(image_bgr, hm.hard, hm.soft, hm.regions, removal_params_of(config));

  FaceInfo finfo;
  finfo.confidence = face.confidence;
  finfo.landmark_count = face.landmarks.empty() ? 0 : face.landmarks.rows;
  finfo.bbox = face.bbox;
  std::vector<std::string> metric_warnings;
  Metrics metrics = compute_metrics(image_bgr, removal.result_bgr, removal.applied_mask,
                                    hm.regions, finfo, config.highlight_removal, &metric_warnings);

  std::vector<std::string> warnings;
  warnings.insert(warnings.end(), hm.regions.warnings.begin(), hm.regions.warnings.end());
  warnings.insert(warnings.end(), hm.warnings.begin(), hm.warnings.end());
  warnings.insert(warnings.end(), removal.warnings.begin(), removal.warnings.end());
  warnings.insert(warnings.end(), metric_warnings.begin(), metric_warnings.end());
  dedupe_warnings(&warnings);

  out.success = true;
  out.status = warnings.empty() ? "处理成功" : "处理完成，存在需要人工复核的警告";
  out.warnings = warnings;
  out.result_bgr = removal.result_bgr;
  out.highlight_mask = hm.hard;
  out.soft_mask = hm.soft;
  out.diff_bgr = removal.diff_bgr;
  out.metrics = metrics;
  out.metrics_text = metrics_to_text(metrics, warnings);
  out.regions = hm.regions;
  out.face = face;
  out.detection_count = detection_count;
  return out;
}

}  // namespace

PipelineOutput process_image(const cv::Mat& image_bgr, const Config& config,
                             const OnnxFaceLandmarker* landmarker,
                             const cv::Mat* injected_landmarks) {
  PipelineOutput fail;
  fail.result_bgr = image_bgr.clone();
  fail.highlight_mask = cv::Mat::zeros(image_bgr.size(), CV_8U);
  fail.soft_mask = cv::Mat::zeros(image_bgr.size(), CV_8U);
  fail.diff_bgr = cv::Mat::zeros(image_bgr.size(), CV_8UC3);

  FaceResult face;
  std::vector<std::string> warnings;
  int detection_count = 0;

  if (injected_landmarks) {
    // 直接使用注入关键点（黄金对照模式），跳过检测。
    face.landmarks = injected_landmarks->clone();
    face.bbox = bbox_from_landmarks(face.landmarks, image_bgr.size());
    face.confidence = 0.90;
    face.detector = "InjectedLandmarks";
    detection_count = 1;
  } else {
    auto [face_scale, hs, ic] = resolve_process_scales(config.pipeline);
    (void)hs;
    (void)ic;
    // 对应 _detect_at_process_scale。
    face_scale = clamp_process_scale(face_scale);
    DetectOutput det;
    if (face_scale >= 0.999) {
      det = detect_face(image_bgr, config.face_detection, landmarker);
    } else {
      cv::Mat small = resize_for_process(image_bgr, face_scale);
      det = detect_face(small, config.face_detection, landmarker);
      if (det.selected) {
        double inv = 1.0 / face_scale;
        det.selected = scale_face_to_shape(*det.selected, inv, image_bgr.size());
        for (auto& f : det.all_faces) f = scale_face_to_shape(f, inv, image_bgr.size());
      }
    }
    if (!det.selected) {
      fail.status = det.status;
      fail.warnings = det.warnings;
      fail.metrics_text = det.status;
      fail.detection_count = static_cast<int>(det.all_faces.size());
      return fail;
    }
    warnings = det.warnings;
    face = *det.selected;
    detection_count = static_cast<int>(det.all_faces.size());
  }

  // 说明：enable_alignment 默认关闭且 CLI 各模式均不启用；C++ 版按原图坐标处理。
  if (config.face_detection.getb("enable_alignment", false))
    warnings.push_back("轻量对齐失败，已回退到原图坐标处理：C++ 版未实现该实验性分支");

  PipelineOutput out = process_core(image_bgr, face, detection_count, config);
  std::vector<std::string> merged = warnings;
  merged.insert(merged.end(), out.warnings.begin(), out.warnings.end());
  dedupe_warnings(&merged);
  out.warnings = merged;
  out.metrics_text = metrics_to_text(out.metrics, out.warnings);
  if (!warnings.empty()) out.status = "处理完成，存在需要人工复核的警告";
  return out;
}

}  // namespace facehi
