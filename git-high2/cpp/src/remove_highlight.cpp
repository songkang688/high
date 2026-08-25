// 对应 highlight_removal/remove_highlight.py。
#include "facehi/remove_highlight.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "facehi/utils.hpp"

namespace facehi {

namespace {

// 对应 _edge_protected_alpha。
//
// numpy 2（NEP 50）dtype 语义：np.clip(edge_strength,0,1) 是强类型 np.float64
// 标量，会把 `alpha * (1.0 - es * edges)` 整条链提升到 float64；
// 而 edge_strength<=0 时直接返回 normalize_mask 的 float32。
// 为了逐位对齐，这里统一用 CV_64F 承载数值，并用 promoted 标记 Python 侧
// 实际 dtype，后续 α 链按该 dtype 选择 float32/float64 舍入路径。
struct AlphaMap {
  cv::Mat a64;    // CV_64F
  bool promoted;  // true：Python 侧为 float64
};

AlphaMap edge_protected_alpha(const cv::Mat& image_bgr, const cv::Mat& soft_mask,
                              double edge_strength) {
  cv::Mat alpha32 = normalize_mask(soft_mask);
  AlphaMap out;
  out.promoted = edge_strength > 0;
  out.a64.create(alpha32.size(), CV_64F);
  if (!out.promoted) {
    for (int y = 0; y < alpha32.rows; ++y) {
      const float* a = alpha32.ptr<float>(y);
      double* o = out.a64.ptr<double>(y);
      for (int x = 0; x < alpha32.cols; ++x) o[x] = static_cast<double>(a[x]);
    }
    return out;
  }
  cv::Mat gray, edges_u8, edges_blur;
  cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Canny(gray, edges_u8, 60, 130);
  cv::GaussianBlur(edges_u8, edges_blur, cv::Size(5, 5), 0);
  double es = std::clamp(edge_strength, 0.0, 1.0);
  for (int y = 0; y < alpha32.rows; ++y) {
    const float* a = alpha32.ptr<float>(y);
    const uint8_t* e = edges_blur.ptr<uint8_t>(y);
    double* o = out.a64.ptr<double>(y);
    for (int x = 0; x < alpha32.cols; ++x) {
      // edges.astype(np.float32)/255.0 先在 float32 舍入，再提升 float64。
      float ev32 = static_cast<float>(e[x]) / 255.0f;
      double v = static_cast<double>(a[x]) * (1.0 - es * static_cast<double>(ev32));
      o[x] = std::clamp(v, 0.0, 1.0);
    }
  }
  return out;
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
//
// dtype 语义严格对齐 numpy 2（NEP 50）：
// - Python 标量（yaml 读出的 float）是弱类型 → 与 float32 数组运算保持 float32；
// - np.clip(标量) / 0.16*np.clip(...) 是 np.float64 强标量 → texture_keep、
//   target_L 以及（edge_strength>0 时）α 链、最终混合都在 float64 里算，
//   只在写回 out_lab(float32) 时舍入一次。
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
  AlphaMap alpha0 = edge_protected_alpha(image_bgr, soft_mask, edge_strength);

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
  // 0.16 * np.clip(ts, 0, 1) → np.float64 强标量，texture_keep 为 float64。
  double tex_gain = 0.16 * std::clamp(texture_strength, 0.0, 1.0);

  double local_sigma = params.getd("local_sigma", 6.0);
  cv::Mat local_skin;
  cv::GaussianBlur(L, local_skin, cv::Size(0, 0), std::max(8.0, local_sigma * 1.6));

  const bool promoted = alpha0.promoted;
  float bs32 = static_cast<float>(brightness_strength);
  float cs32 = static_cast<float>(color_strength);
  float fa32 = static_cast<float>(final_alpha);
  float lf32 = static_cast<float>(luminance_floor);

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
    const double* pa0 = alpha0.a64.ptr<double>(y);
    cv::Vec3f* po = out_lab.ptr<cv::Vec3f>(y);
    for (int x = 0; x < w; ++x) {
      // texture_keep = (L - low)(f32) * tex_gain(f64) → float64
      float texture32 = pL[x] - plow[x];
      double texture_keep = static_cast<double>(texture32) * tex_gain;
      // min_allowed = max(L*lf, ls*0.93)：弱标量 → float32
      float min_allowed32 = std::max(pL[x] * lf32, pls[x] * 0.93f);
      // target_L：float64（texture_keep 提升所致）
      double target_L = std::max(static_cast<double>(prL[x]) + texture_keep,
                                 static_cast<double>(min_allowed32));
      target_L = std::max(target_L, static_cast<double>(pls[x] - 5.0f));
      target_L = std::min(target_L, static_cast<double>(pL[x] + 0.5f));

      if (promoted) {
        // α 链 float64：alpha0(f64) * 弱标量 → float64
        double alpha_l = std::clamp(pa0[x] * brightness_strength * final_alpha, 0.0, 1.0);
        double alpha_c = std::clamp(pa0[x] * color_strength * final_alpha, 0.0, 1.0);
        po[x][0] = static_cast<float>(static_cast<double>(pL[x]) * (1.0 - alpha_l) +
                                      target_L * alpha_l);
        po[x][1] = static_cast<float>(static_cast<double>(pA[x]) * (1.0 - alpha_c) +
                                      static_cast<double>(prA[x]) * alpha_c);
        po[x][2] = static_cast<float>(static_cast<double>(pB[x]) * (1.0 - alpha_c) +
                                      static_cast<double>(prB[x]) * alpha_c);
      } else {
        // α 链 float32；但 target_L 仍是 float64，L*(1-αl)+target_L*αl 提升 float64
        float a32 = static_cast<float>(pa0[x]);
        float alpha_l = std::clamp(a32 * bs32 * fa32, 0.0f, 1.0f);
        float alpha_c = std::clamp(a32 * cs32 * fa32, 0.0f, 1.0f);
        double part_keep = static_cast<double>(pL[x] * (1.0f - alpha_l));
        po[x][0] = static_cast<float>(part_keep + target_L * static_cast<double>(alpha_l));
        po[x][1] = pA[x] * (1.0f - alpha_c) + prA[x] * alpha_c;
        po[x][2] = pB[x] * (1.0f - alpha_c) + prB[x] * alpha_c;
      }
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
    const double* pa0 = alpha0.a64.ptr<double>(y);
    const cv::Vec3b* src = image_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b* mod = out_bgr.ptr<cv::Vec3b>(y);
    cv::Vec3b* po = result.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      // keep = alpha0 <= 0.001：promoted 时 float64 比较，否则 float32 比较。
      bool keep = promoted ? (pa0[x] <= 0.001)
                           : (static_cast<float>(pa0[x]) <= 0.001f);
      po[x] = keep ? src[x] : mod[x];
    }
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
