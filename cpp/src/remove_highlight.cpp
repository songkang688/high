// 对应 highlight_removal/remove_highlight.py。
#include "facehi/remove_highlight.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "facehi/utils.hpp"

namespace facehi {

namespace {

// 对应 _edge_protected_alpha：返回 float32 [0,1]。
cv::Mat edge_protected_alpha(const cv::Mat& image_bgr, const cv::Mat& soft_mask,
                             double edge_strength) {
  cv::Mat alpha = normalize_mask(soft_mask);
  if (edge_strength <= 0) return alpha;
  cv::Mat gray, edges_u8, edges_blur;
  cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Canny(gray, edges_u8, 60, 130);
  cv::GaussianBlur(edges_u8, edges_blur, cv::Size(5, 5), 0);
  float es = static_cast<float>(std::clamp(edge_strength, 0.0, 1.0));
  for (int y = 0; y < alpha.rows; ++y) {
    float* a = alpha.ptr<float>(y);
    const uint8_t* e = edges_blur.ptr<uint8_t>(y);
    for (int x = 0; x < alpha.cols; ++x) {
      float ev = static_cast<float>(e[x]) / 255.0f;
      a[x] = std::clamp(a[x] * (1.0f - es * ev), 0.0f, 1.0f);
    }
  }
  return alpha;
}

// 对应 _inpaint_lab_reference：返回 float32 3 通道 Lab。
cv::Mat inpaint_lab_reference(const cv::Mat& image_bgr, const cv::Mat& hard_mask, int radius) {
  radius = std::max(1, std::min(9, radius));
  cv::Mat lab;
  cv::cvtColor(image_bgr, lab, cv::COLOR_BGR2Lab);
  if (cv::countNonZero(hard_mask) == 0) {
    cv::Mat labf;
    lab.convertTo(labf, CV_32F);
    return labf;
  }
  std::vector<cv::Mat> chans;
  cv::split(lab, chans);
  std::vector<cv::Mat> filled(3);
  for (int c = 0; c < 3; ++c)
    cv::inpaint(chans[c], hard_mask, filled[c], radius, cv::INPAINT_TELEA);
  cv::Mat ref_u8, ref;
  cv::merge(filled, ref_u8);
  ref_u8.convertTo(ref, CV_32F);

  cv::Mat bilateral, ref2_u8, ref2;
  cv::bilateralFilter(image_bgr, bilateral, 11, 45, 45);
  cv::cvtColor(bilateral, ref2_u8, cv::COLOR_BGR2Lab);
  ref2_u8.convertTo(ref2, CV_32F);

  cv::Mat out(ref.size(), CV_32FC3);
  for (int y = 0; y < out.rows; ++y) {
    const cv::Vec3f* r1 = ref.ptr<cv::Vec3f>(y);
    const cv::Vec3f* r2 = ref2.ptr<cv::Vec3f>(y);
    cv::Vec3f* o = out.ptr<cv::Vec3f>(y);
    for (int x = 0; x < out.cols; ++x)
      for (int c = 0; c < 3; ++c) o[x][c] = 0.72f * r1[x][c] + 0.28f * r2[x][c];
  }
  return out;
}

// 对应 _faithful_suppress。
cv::Mat faithful_suppress(const cv::Mat& image_bgr, const cv::Mat& hard_mask,
                          const cv::Mat& soft_mask, const Params& params) {
  double brightness_strength = params.getd("brightness_suppress_strength", 0.74);
  double color_strength = params.getd("chroma_restore_strength", 0.30);
  double texture_strength = params.getd("texture_preserve_strength", 0.78);
  double edge_strength = params.getd("edge_protect_strength", 0.45);
  double final_alpha = params.getd("final_blend_alpha", 0.92);
  int radius = params.geti("inpainting_radius", 3);
  double luminance_floor = params.getd("faithful_luminance_floor", 0.86);

  const int h = image_bgr.rows, w = image_bgr.cols;
  cv::Mat alpha0 = edge_protected_alpha(image_bgr, soft_mask, edge_strength);

  cv::Mat lab_u8, lab;
  cv::cvtColor(image_bgr, lab_u8, cv::COLOR_BGR2Lab);
  lab_u8.convertTo(lab, CV_32F);
  cv::Mat ref_lab = inpaint_lab_reference(image_bgr, hard_mask, radius);

  std::vector<cv::Mat> lab_ch(3), ref_ch(3);
  cv::split(lab, lab_ch);
  cv::split(ref_lab, ref_ch);
  cv::Mat& L = lab_ch[0];
  cv::Mat& ref_L = ref_ch[0];

  cv::Mat low;
  cv::GaussianBlur(L, low, cv::Size(0, 0), 2.2);
  // Python 先在双精度算 0.16*clip(ts)，再与 float32 数组相乘。
  float tex_gain = static_cast<float>(0.16 * std::clamp(texture_strength, 0.0, 1.0));

  double local_sigma = params.getd("local_sigma", 6.0);
  cv::Mat local_skin;
  cv::GaussianBlur(L, local_skin, cv::Size(0, 0), std::max(8.0, local_sigma * 1.6));

  float bs = static_cast<float>(brightness_strength);
  float cs = static_cast<float>(color_strength);
  float fa = static_cast<float>(final_alpha);
  float lf = static_cast<float>(luminance_floor);

  cv::Mat out_lab(h, w, CV_32FC3);
  for (int y = 0; y < h; ++y) {
    const float* pL = L.ptr<float>(y);
    const float* pA = lab_ch[1].ptr<float>(y);
    const float* pB = lab_ch[2].ptr<float>(y);
    const float* prL = ref_L.ptr<float>(y);
    const float* prA = ref_ch[1].ptr<float>(y);
    const float* prB = ref_ch[2].ptr<float>(y);
    const float* plow = low.ptr<float>(y);
    const float* pls = local_skin.ptr<float>(y);
    const float* pa0 = alpha0.ptr<float>(y);
    cv::Vec3f* po = out_lab.ptr<cv::Vec3f>(y);
    for (int x = 0; x < w; ++x) {
      float alpha_l = std::clamp(pa0[x] * bs * fa, 0.0f, 1.0f);
      float alpha_c = std::clamp(pa0[x] * cs * fa, 0.0f, 1.0f);
      float texture = pL[x] - plow[x];
      float texture_keep = texture * tex_gain;
      float min_allowed = std::max(pL[x] * lf, pls[x] * 0.93f);
      float target_L = std::max(prL[x] + texture_keep, min_allowed);
      target_L = std::max(target_L, pls[x] - 5.0f);
      target_L = std::min(target_L, pL[x] + 0.5f);
      po[x][0] = pL[x] * (1.0f - alpha_l) + target_L * alpha_l;
      po[x][1] = pA[x] * (1.0f - alpha_c) + prA[x] * alpha_c;
      po[x][2] = pB[x] * (1.0f - alpha_c) + prB[x] * alpha_c;
    }
  }

  // np.clip(out_lab, 0, 255).astype(np.uint8) → 截断取整。
  cv::Mat out_lab_u8(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y) {
    const cv::Vec3f* pi = out_lab.ptr<cv::Vec3f>(y);
    cv::Vec3b* po = out_lab_u8.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c)
        po[x][c] = static_cast<uint8_t>(std::clamp(pi[x][c], 0.0f, 255.0f));
  }
  cv::Mat out_bgr;
  cv::cvtColor(out_lab_u8, out_bgr, cv::COLOR_Lab2BGR);

  cv::Mat result(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y) {
    const float* pa0 = alpha0.ptr<float>(y);
    const cv::Vec3b* src = image_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b* mod = out_bgr.ptr<cv::Vec3b>(y);
    cv::Vec3b* po = result.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) po[x] = (pa0[x] <= 0.001f) ? src[x] : mod[x];
  }
  return result;
}

// 对应 _strong_inpaint。
cv::Mat strong_inpaint(const cv::Mat& image_bgr, const cv::Mat& hard_mask,
                       const cv::Mat& soft_mask, const Params& params,
                       const cv::Mat* base_image = nullptr) {
  const cv::Mat& source = base_image ? *base_image : image_bgr;
  int radius = std::max(1, std::min(9, params.geti("inpainting_radius", 3)));
  double alpha_strength = params.getd("poisson_alpha_strength", 0.42);
  double final_alpha = params.getd("final_blend_alpha", 0.92);
  if (cv::countNonZero(hard_mask) == 0) return source.clone();
  cv::Mat inpainted;
  cv::inpaint(source, hard_mask, inpainted, radius, cv::INPAINT_TELEA);
  cv::Mat alpha = normalize_mask(soft_mask);
  float as = static_cast<float>(alpha_strength);
  float fa = static_cast<float>(final_alpha);
  cv::Mat out(source.size(), CV_8UC3);
  for (int y = 0; y < source.rows; ++y) {
    const float* pa = alpha.ptr<float>(y);
    const cv::Vec3b* ps = source.ptr<cv::Vec3b>(y);
    const cv::Vec3b* pi = inpainted.ptr<cv::Vec3b>(y);
    cv::Vec3b* po = out.ptr<cv::Vec3b>(y);
    for (int x = 0; x < source.cols; ++x) {
      float a = std::clamp(pa[x] * as * fa, 0.0f, 1.0f);
      for (int c = 0; c < 3; ++c) {
        float v = static_cast<float>(ps[x][c]) * (1.0f - a) + static_cast<float>(pi[x][c]) * a;
        po[x][c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));  // 截断取整
      }
    }
  }
  return out;
}

cv::Mat diff_vis_of(const cv::Mat& image_bgr, const cv::Mat& result) {
  cv::Mat diff;
  cv::absdiff(image_bgr, result, diff);
  cv::Mat out(diff.size(), CV_8UC3);
  for (int y = 0; y < diff.rows; ++y) {
    const cv::Vec3b* d = diff.ptr<cv::Vec3b>(y);
    cv::Vec3b* o = out.ptr<cv::Vec3b>(y);
    for (int x = 0; x < diff.cols; ++x)
      for (int c = 0; c < 3; ++c)
        o[x][c] = static_cast<uint8_t>(std::clamp(static_cast<float>(d[x][c]) * 4.0f, 0.0f, 255.0f));
  }
  return out;
}

RemovalOutput remove_highlight_core(const cv::Mat& image_bgr, const cv::Mat& hard_mask,
                                    const cv::Mat& soft_mask, const RegionMasks& regions,
                                    const Params& params) {
  RemovalOutput out;
  std::string mode = params.gets("mode", "保真");
  const cv::Mat& skin_u8 = regions.masks.at("skin");
  const cv::Mat& protect_u8 = regions.masks.at("protect");
  const int h = image_bgr.rows, w = image_bgr.cols;

  cv::Mat hard = hard_mask.clone();
  cv::Mat soft = soft_mask.clone();
  for (int y = 0; y < h; ++y) {
    const uint8_t* sk = skin_u8.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    uint8_t* ph = hard.ptr<uint8_t>(y);
    uint8_t* ps = soft.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      if (sk[x] == 0 || pr[x] > 0) {
        ph[x] = 0;
        ps[x] = 0;
      }
    }
  }

  long long face_area = std::max<long long>(1, cv::countNonZero(regions.masks.at("face_mask")));
  double mod_area_ratio = static_cast<double>(cv::countNonZero(hard)) / static_cast<double>(face_area);
  double max_allowed = params.getd("max_allowed_modify_area_ratio", 0.15);
  if (mod_area_ratio > max_allowed)
    out.warnings.push_back("当前修改区域过大，可能影响真实性，请降低处理强度");

  cv::Mat result;
  if (mode == "强修复") {
    result = strong_inpaint(image_bgr, hard, soft, params);
  } else if (mode == "混合") {
    cv::Mat faithful = faithful_suppress(image_bgr, hard, soft, params);
    cv::Mat lab;
    cv::cvtColor(image_bgr, lab, cv::COLOR_BGR2Lab);
    double extra = params.getd("extreme_core_extra_l", 12);
    double lab_l_thr = params.getd("lab_l_threshold", 188);
    float extreme_thr = static_cast<float>(lab_l_thr + extra);
    cv::Mat extreme(h, w, CV_8U);
    long long n_extreme = 0;
    for (int y = 0; y < h; ++y) {
      const uint8_t* ph = hard.ptr<uint8_t>(y);
      const cv::Vec3b* pl = lab.ptr<cv::Vec3b>(y);
      uint8_t* pe = extreme.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) {
        pe[x] = (ph[x] > 0 && static_cast<float>(pl[x][0]) >= extreme_thr) ? 255 : 0;
        n_extreme += pe[x] ? 1 : 0;
      }
    }
    if (n_extreme > 0) {
      double sigma = std::max(1.0, params.getd("mask_blur_radius", 9) / 2.5);
      cv::Mat extreme_soft;
      cv::GaussianBlur(extreme, extreme_soft, cv::Size(0, 0), sigma);
      result = strong_inpaint(image_bgr, extreme, extreme_soft, params, &faithful);
    } else {
      result = faithful;
    }
  } else {
    result = faithful_suppress(image_bgr, hard, soft, params);
  }

  // 强制保护：五官、背景、非皮肤区域回写原图。
  for (int y = 0; y < h; ++y) {
    const uint8_t* ps = soft.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    const uint8_t* sk = skin_u8.ptr<uint8_t>(y);
    const cv::Vec3b* src = image_bgr.ptr<cv::Vec3b>(y);
    cv::Vec3b* po = result.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x)
      if (ps[x] == 0 || pr[x] > 0 || sk[x] == 0) po[x] = src[x];
  }

  out.result_bgr = result;
  out.applied_mask = hard;
  out.diff_bgr = diff_vis_of(image_bgr, result);
  return out;
}

}  // namespace

RemovalOutput remove_highlight(const cv::Mat& image_bgr, const cv::Mat& hard_mask,
                               const cv::Mat& soft_mask, const RegionMasks& regions,
                               const Params& params) {
  bool enable_roi = params.getb("enable_roi_removal", true);
  double padding_ratio = params.getd("roi_padding_ratio", 0.12);
  if (enable_roi) {
    auto roi = compute_roi_bbox(hard_mask, soft_mask, image_bgr.size(), padding_ratio);
    if (roi) {
      cv::Rect r = *roi;
      if (r.width < image_bgr.cols || r.height < image_bgr.rows) {
        RegionMasks cropped;
        cropped.geometry = regions.geometry;
        cropped.warnings = regions.warnings;
        for (const auto& [k, m] : regions.masks) cropped.masks[k] = m(r);
        RemovalOutput core = remove_highlight_core(image_bgr(r), hard_mask(r), soft_mask(r),
                                                   cropped, params);
        RemovalOutput out;
        out.result_bgr = image_bgr.clone();
        core.result_bgr.copyTo(out.result_bgr(r));
        out.applied_mask = hard_mask.clone();
        out.diff_bgr = diff_vis_of(image_bgr, out.result_bgr);
        out.warnings = core.warnings;
        return out;
      }
    }
  }
  return remove_highlight_core(image_bgr, hard_mask, soft_mask, regions, params);
}

}  // namespace facehi
