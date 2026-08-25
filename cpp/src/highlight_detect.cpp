// 对应 highlight_removal/highlight_detect.py。
#include "facehi/highlight_detect.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "facehi/utils.hpp"

namespace facehi {

namespace {

cv::Mat remove_small_components(const cv::Mat& mask_u8, int min_area) {
  cv::Mat m = mask_u8 > 0;  // 0/255
  cv::Mat bin;
  cv::threshold(m, bin, 0, 1, cv::THRESH_BINARY);
  cv::Mat labels, stats, centroids;
  int num = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
  cv::Mat out = cv::Mat::zeros(mask_u8.size(), CV_8U);
  std::vector<uint8_t> keep(static_cast<size_t>(num), 0);
  for (int i = 1; i < num; ++i)
    if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area) keep[static_cast<size_t>(i)] = 1;
  for (int y = 0; y < out.rows; ++y) {
    const int* lb = labels.ptr<int>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    for (int x = 0; x < out.cols; ++x)
      if (keep[static_cast<size_t>(lb[x])]) o[x] = 255;
  }
  return out;
}

// 对应 _masked_gaussian。
cv::Mat masked_gaussian(const cv::Mat& values32f, const cv::Mat& valid_bool8, double sigma) {
  cv::Mat valid_f(valid_bool8.size(), CV_32F);
  for (int y = 0; y < valid_bool8.rows; ++y) {
    const uint8_t* v = valid_bool8.ptr<uint8_t>(y);
    float* o = valid_f.ptr<float>(y);
    for (int x = 0; x < valid_bool8.cols; ++x) o[x] = v[x] ? 1.0f : 0.0f;
  }
  cv::Mat prod(values32f.size(), CV_32F);
  for (int y = 0; y < values32f.rows; ++y) {
    const float* a = values32f.ptr<float>(y);
    const float* b = valid_f.ptr<float>(y);
    float* o = prod.ptr<float>(y);
    for (int x = 0; x < values32f.cols; ++x) o[x] = a[x] * b[x];
  }
  cv::Mat num, den;
  cv::GaussianBlur(prod, num, cv::Size(0, 0), sigma, sigma);
  cv::GaussianBlur(valid_f, den, cv::Size(0, 0), sigma, sigma);
  cv::Mat out(values32f.size(), CV_32F);
  for (int y = 0; y < out.rows; ++y) {
    const float* n = num.ptr<float>(y);
    const float* d = den.ptr<float>(y);
    float* o = out.ptr<float>(y);
    for (int x = 0; x < out.cols; ++x) o[x] = n[x] / std::max(d[x], 1e-3f);
  }
  return out;
}

// 对应 _grow_from_core。
cv::Mat grow_from_core(const cv::Mat& core_bool8, const cv::Mat& halo_bool8, int max_iter) {
  cv::Mat grown = core_bool8.clone();
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  for (int it = 0; it < std::max(0, max_iter); ++it) {
    cv::Mat dil;
    cv::dilate(grown, dil, kernel);
    cv::Mat nxt(grown.size(), CV_8U);
    for (int y = 0; y < grown.rows; ++y) {
      const uint8_t* d = dil.ptr<uint8_t>(y);
      const uint8_t* h = halo_bool8.ptr<uint8_t>(y);
      uint8_t* o = nxt.ptr<uint8_t>(y);
      for (int x = 0; x < grown.cols; ++x) o[x] = (d[x] && h[x]) ? 1 : 0;
    }
    if (cv::countNonZero(nxt != grown) == 0) break;
    grown = nxt;
  }
  return grown;
}

// 对应 _cap_by_score。
cv::Mat cap_by_score(const cv::Mat& mask_bool8, const cv::Mat& score32f,
                     const cv::Mat& region_bool8, long long max_pixels) {
  cv::Mat out(mask_bool8.size(), CV_8U);
  long long count = 0;
  for (int y = 0; y < out.rows; ++y) {
    const uint8_t* m = mask_bool8.ptr<uint8_t>(y);
    const uint8_t* r = region_bool8.ptr<uint8_t>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    for (int x = 0; x < out.cols; ++x) {
      o[x] = (m[x] && r[x]) ? 1 : 0;
      count += o[x];
    }
  }
  if (max_pixels <= 0 || count <= max_pixels) return out;
  std::vector<float> vals = masked_values(score32f, out);
  if (static_cast<long long>(vals.size()) <= max_pixels) return out;
  std::nth_element(vals.begin(), vals.end() - max_pixels, vals.end());
  float kth = vals[vals.size() - static_cast<size_t>(max_pixels)];
  for (int y = 0; y < out.rows; ++y) {
    const float* s = score32f.ptr<float>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    for (int x = 0; x < out.cols; ++x)
      if (o[x] && !(s[x] >= kth)) o[x] = 0;
  }
  return out;
}

double region_fraction(const std::string& key, const Params& p) {
  if (key == "forehead") return p.getd("forehead_region_max_fraction", 0.30);
  if (key == "nose_bridge") return p.getd("nose_bridge_region_max_fraction", 0.72);
  if (key == "nose_tip") return p.getd("nose_tip_region_max_fraction", 0.55);
  if (key == "left_cheek" || key == "right_cheek") return p.getd("cheek_region_max_fraction", 0.20);
  if (key == "chin") return p.getd("chin_region_max_fraction", 0.24);
  if (key == "philtrum") return p.getd("philtrum_region_max_fraction", 0.32);
  if (key == "left_brow_ridge" || key == "right_brow_ridge")
    return p.getd("brow_region_max_fraction", 0.16);
  if (key == "highlight_candidate" || key == "treatable_skin")
    return p.getd("skin_region_max_fraction", 0.35);
  return 0.20;
}

struct AdaptiveResult {
  cv::Mat core;   // CV_8U 0/1
  cv::Mat grown;  // CV_8U 0/1
  cv::Mat score;  // CV_32F
};

AdaptiveResult adaptive_region_mask(
    const std::string& key, const cv::Mat& region_mask_u8, const cv::Mat& base_bool8,
    const cv::Mat& rgb_max, const cv::Mat& rgb_range, const cv::Mat& gray, const cv::Mat& v,
    const cv::Mat& s, const cv::Mat& l, const cv::Mat& chroma_delta, const cv::Mat& local_diff,
    const cv::Mat& large_diff, const Params& params) {
  const int h = base_bool8.rows, w = base_bool8.cols;
  AdaptiveResult res;
  res.core = cv::Mat::zeros(h, w, CV_8U);
  res.grown = cv::Mat::zeros(h, w, CV_8U);
  res.score = cv::Mat::zeros(h, w, CV_32F);

  cv::Mat region(h, w, CV_8U);
  long long region_count = 0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* r = region_mask_u8.ptr<uint8_t>(y);
    const uint8_t* b = base_bool8.ptr<uint8_t>(y);
    uint8_t* o = region.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      o[x] = (r[x] > 0 && b[x]) ? 1 : 0;
      region_count += o[x];
    }
  }
  if (region_count < 20) return res;

  std::vector<float> lv = masked_values(l, region);
  std::vector<float> vv = masked_values(v, region);
  std::vector<float> sv = masked_values(s, region);
  std::vector<float> gv = masked_values(gray, region);
  std::vector<float> cdv = masked_values(chroma_delta, region);
  std::vector<float> ldv = masked_values(local_diff, region);

  double core_pct = params.getd("adaptive_core_percentile", 90);
  double l50 = percentile_or(lv, 50, 185);
  double l75 = percentile_or(lv, 75, l50 + 7);
  double l90 = percentile_or(lv, core_pct, l50 + 14);
  double l95 = percentile_or(lv, 95, l90 + 5);
  double v50 = percentile_or(vv, 50, 205);
  double v75 = percentile_or(vv, 75, v50 + 7);
  double v90 = percentile_or(vv, 90, v50 + 14);
  double s50 = percentile_or(sv, 50, 95);
  double s75 = percentile_or(sv, 75, s50 + 10);
  double g50 = percentile_or(gv, 50, 175);
  double g75 = percentile_or(gv, 75, g50 + 7);
  double g90 = percentile_or(gv, 90, g50 + 14);
  double cd75 = percentile_or(cdv, 75, 24);
  double ld75 = percentile_or(ldv, 75, 2);
  double ld90 = percentile_or(ldv, 90, 5);

  double abs_l_thr = params.getd({"lab_l_threshold", "lab_l_thr"}, 188);
  double abs_v_thr = params.getd({"hsv_v_threshold", "hsv_v_thr"}, 190);
  double rgb_thr = params.getd({"rgb_brightness_threshold", "rgb_thr"}, 205);
  double hsv_s_upper = params.getd("hsv_s_upper", 150);
  double oil_s_upper = params.getd("oil_shine_s_upper", std::max(150.0, hsv_s_upper));
  double delta = params.getd("adaptive_region_delta", 4.8);
  double core_delta = params.getd("adaptive_core_delta", 8.0);
  double rel_sat_drop = params.getd("relative_saturation_drop", -8.0);
  double halo_pct = params.getd("adaptive_halo_percentile", 78);

  double l_halo_thr = std::max(percentile_or(lv, halo_pct, l75), l50 + delta);
  double l_core_thr = std::max(percentile_or(lv, core_pct, l90), l50 + core_delta);
  double v_halo_thr = std::max(percentile_or(vv, halo_pct, v75), v50 + delta);
  double g_halo_thr = std::max(percentile_or(gv, halo_pct, g75), g50 + delta);

  if (key == "nose_bridge" || key == "nose_tip") {
    l_halo_thr -= 2.0;
    v_halo_thr -= 2.0;
    g_halo_thr -= 2.0;
    l_core_thr -= 2.0;
  } else if (key == "left_brow_ridge" || key == "right_brow_ridge") {
    l_halo_thr -= 2.0;
    v_halo_thr -= 2.0;
    g_halo_thr -= 2.0;
    l_core_thr -= 2.5;
  } else if (key == "left_cheek" || key == "right_cheek") {
    l_halo_thr += 1.5;
    l_core_thr += 1.0;
  } else if (key == "chin") {
    l_halo_thr += 1.0;
  }

  double local_thr = params.getd("local_brightness_threshold", 5);
  double chroma_upper = params.getd("local_skin_chroma_delta_upper", 42);
  double sat_pix_thr = params.getd("saturation_pixel_threshold", 245);

  // 归一化分母（双精度，随后按 float32 语义使用）。
  float l_den = static_cast<float>(std::max(l95 - l50, 8.0));
  float v_den = static_cast<float>(std::max(v90 - v50, 8.0));
  float g_den = static_cast<float>(std::max(g90 - g50, 8.0));
  float local_den = static_cast<float>(std::max(std::max(ld90, ld75 + 1.0), 3.5));
  float s_white_thr = static_cast<float>(std::max(hsv_s_upper, s50 + 30));
  float chroma_thr14 = static_cast<float>(std::max(chroma_upper, cd75 + 14));
  float chroma_thr18 = static_cast<float>(std::max(chroma_upper, cd75 + 18.0));
  float warm_thr = static_cast<float>(std::max(oil_s_upper, s75 + 42.0));
  float rel_sat_thr = static_cast<float>(s50 - rel_sat_drop);
  float local_core_thr = static_cast<float>(std::max(local_thr, ld75 + 0.5));
  float large_halo_thr = static_cast<float>(std::max(2.0, local_thr * 0.50));

  cv::Mat core(h, w, CV_8U, cv::Scalar(0));
  cv::Mat halo(h, w, CV_8U, cv::Scalar(0));
  cv::Mat& score = res.score;
  long long core_count = 0;

  for (int y = 0; y < h; ++y) {
    const uint8_t* rg = region.ptr<uint8_t>(y);
    const float* pl = l.ptr<float>(y);
    const float* pv = v.ptr<float>(y);
    const float* ps = s.ptr<float>(y);
    const float* pg = gray.ptr<float>(y);
    const float* prm = rgb_max.ptr<float>(y);
    const float* prr = rgb_range.ptr<float>(y);
    const float* pcd = chroma_delta.ptr<float>(y);
    const float* pld = local_diff.ptr<float>(y);
    const float* plg = large_diff.ptr<float>(y);
    float* psc = score.ptr<float>(y);
    uint8_t* pco = core.ptr<uint8_t>(y);
    uint8_t* pha = halo.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      // 得分对全图计算（Python 中 score 是全图数组）。
      float l_norm = (pl[x] - static_cast<float>(l50)) / l_den;
      float v_norm = (pv[x] - static_cast<float>(v50)) / v_den;
      float g_norm = (pg[x] - static_cast<float>(g50)) / g_den;
      float local_norm = std::max(pld[x], plg[x] * 0.65f) / local_den;
      float white_score = (prr[x] <= 62 ? 1.0f : 0.0f) * 0.55f + (ps[x] <= s_white_thr ? 1.0f : 0.0f) * 0.25f;
      float abs_score = ((pl[x] >= static_cast<float>(abs_l_thr) ? 1.0f : 0.0f) +
                         (pv[x] >= static_cast<float>(abs_v_thr) ? 1.0f : 0.0f) +
                         (prm[x] >= static_cast<float>(rgb_thr) ? 1.0f : 0.0f)) /
                        3.0f;
      float chroma_score = pcd[x] <= chroma_thr14 ? 1.0f : 0.0f;
      float sc = 0.40f * std::clamp(l_norm, 0.0f, 2.0f);
      sc = sc + 0.22f * std::clamp(v_norm, 0.0f, 2.0f);
      sc = sc + 0.13f * std::clamp(g_norm, 0.0f, 2.0f);
      sc = sc + 0.16f * std::clamp(local_norm, 0.0f, 2.0f);
      sc = sc + 0.12f * white_score;
      sc = sc + 0.18f * abs_score;
      sc = sc + 0.08f * chroma_score;
      psc[x] = sc;

      bool warm_allowed = ps[x] <= warm_thr;
      bool not_too_colored = pcd[x] <= chroma_thr18;
      bool relative_low_sat = ps[x] <= rel_sat_thr;

      bool absolute_core =
          ((pl[x] >= static_cast<float>(abs_l_thr) && pv[x] >= static_cast<float>(abs_v_thr - 4) && warm_allowed) ||
           (prm[x] >= static_cast<float>(rgb_thr) && pl[x] >= static_cast<float>(abs_l_thr - 8) && warm_allowed) ||
           (prm[x] >= static_cast<float>(sat_pix_thr - 3) && prr[x] <= 92));
      bool percentile_core = rg[x] && pl[x] >= static_cast<float>(l_core_thr) &&
                             pv[x] >= static_cast<float>(v_halo_thr - 2) && warm_allowed;
      bool local_core = rg[x] && pld[x] >= local_core_thr &&
                        pl[x] >= static_cast<float>(l_halo_thr - 2) && warm_allowed;
      bool c = rg[x] && not_too_colored &&
               (absolute_core || percentile_core || local_core || (sc >= 0.74f && relative_low_sat));
      pco[x] = c ? 1 : 0;
      core_count += pco[x];

      bool ha = rg[x] && not_too_colored && warm_allowed &&
                (pl[x] >= static_cast<float>(l_halo_thr) || pv[x] >= static_cast<float>(v_halo_thr) ||
                 pg[x] >= static_cast<float>(g_halo_thr) ||
                 (sc >= 0.46f && pl[x] >= static_cast<float>(l50 + delta * 0.55)) ||
                 (plg[x] >= large_halo_thr && pl[x] >= static_cast<float>(l50 + delta * 0.5)));
      pha[x] = ha ? 1 : 0;
    }
  }

  // 平滑油光 fallback 种子。
  if (core_count == 0) {
    std::vector<float> rmv = masked_values(rgb_max, region);
    double rm95 = percentile_or(rmv, 95, 0);
    if (l95 >= abs_l_thr - 5 || v90 >= abs_v_thr - 4 || rm95 >= rgb_thr - 4) {
      double seed_pct = params.getd("fallback_seed_percentile", 93);
      std::vector<float> scv = masked_values(score, region);
      double seed_thr = percentile_or(scv, seed_pct, 0.85);
      for (int y = 0; y < h; ++y) {
        const uint8_t* rg = region.ptr<uint8_t>(y);
        const float* psc = score.ptr<float>(y);
        const float* pl = l.ptr<float>(y);
        const float* ps = s.ptr<float>(y);
        const float* pcd = chroma_delta.ptr<float>(y);
        uint8_t* pco = core.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
          bool c = rg[x] && pcd[x] <= chroma_thr18 && psc[x] >= static_cast<float>(seed_thr) &&
                   pl[x] >= static_cast<float>(l_halo_thr - 1) && ps[x] <= warm_thr;
          pco[x] = c ? 1 : 0;
        }
      }
    }
  }

  cv::Mat grown = grow_from_core(core, halo, params.geti("adaptive_grow_iterations", 12));
  double max_fraction = std::clamp(region_fraction(key, params), 0.03, 0.95);
  long long max_pixels = std::max<long long>(6, static_cast<long long>(region_count * max_fraction));
  cv::Mat grown_or_core(h, w, CV_8U);
  for (int y = 0; y < h; ++y) {
    const uint8_t* g = grown.ptr<uint8_t>(y);
    const uint8_t* c = core.ptr<uint8_t>(y);
    uint8_t* o = grown_or_core.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) o[x] = (g[x] || c[x]) ? 1 : 0;
  }
  res.grown = cap_by_score(grown_or_core, score, region, max_pixels);
  res.core = cap_by_score(core, score, region,
                          std::max<long long>(3, static_cast<long long>(max_pixels * 0.55)));
  return res;
}

}  // namespace

HighlightOutput detect_highlight(const cv::Mat& image_bgr, const RegionMasks& regions,
                                 const Params& params) {
  HighlightOutput out;
  const int h = image_bgr.rows, w = image_bgr.cols;

  auto get_mask = [&](const std::string& key) -> cv::Mat {
    auto it = regions.masks.find(key);
    if (it == regions.masks.end()) return cv::Mat::zeros(h, w, CV_8U);
    return it->second;
  };
  cv::Mat skin_u8 = get_mask("skin");
  cv::Mat protect_u8 = get_mask("protect");
  cv::Mat candidate_u8 = get_mask("highlight_candidate");

  cv::Mat base(h, w, CV_8U);
  long long base_count = 0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* sk = skin_u8.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    const uint8_t* ca = candidate_u8.ptr<uint8_t>(y);
    uint8_t* b = base.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      b[x] = (sk[x] > 0 && ca[x] > 0 && pr[x] == 0) ? 1 : 0;
      base_count += b[x];
    }
  }
  if (base_count == 0) {
    out.hard_mask = cv::Mat::zeros(h, w, CV_8U);
    out.soft_mask = cv::Mat::zeros(h, w, CV_8U);
    out.warnings.push_back("高光候选区域为空，未执行高光检测");
    return out;
  }

  cv::Mat hsv, lab, gray_u8;
  cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(image_bgr, lab, cv::COLOR_BGR2Lab);
  cv::cvtColor(image_bgr, gray_u8, cv::COLOR_BGR2GRAY);

  cv::Mat rgb_max(h, w, CV_32F), rgb_range(h, w, CV_32F), gray(h, w, CV_32F);
  cv::Mat v(h, w, CV_32F), s(h, w, CV_32F), l(h, w, CV_32F), a(h, w, CV_32F), b(h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b* im = image_bgr.ptr<cv::Vec3b>(y);
    const cv::Vec3b* hv = hsv.ptr<cv::Vec3b>(y);
    const cv::Vec3b* lb = lab.ptr<cv::Vec3b>(y);
    const uint8_t* gr = gray_u8.ptr<uint8_t>(y);
    float* prm = rgb_max.ptr<float>(y);
    float* prr = rgb_range.ptr<float>(y);
    float* pg = gray.ptr<float>(y);
    float* pv = v.ptr<float>(y);
    float* ps = s.ptr<float>(y);
    float* pl = l.ptr<float>(y);
    float* pa = a.ptr<float>(y);
    float* pb = b.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      // BGR → RGB 的 max/min 与通道顺序无关。
      uint8_t mx = std::max({im[x][0], im[x][1], im[x][2]});
      uint8_t mn = std::min({im[x][0], im[x][1], im[x][2]});
      prm[x] = static_cast<float>(mx);
      prr[x] = static_cast<float>(mx) - static_cast<float>(mn);
      pg[x] = static_cast<float>(gr[x]);
      pv[x] = static_cast<float>(hv[x][2]);
      ps[x] = static_cast<float>(hv[x][1]);
      pl[x] = static_cast<float>(lb[x][0]);
      pa[x] = static_cast<float>(lb[x][1]);
      pb[x] = static_cast<float>(lb[x][2]);
    }
  }

  double rgb_thr = params.getd({"rgb_brightness_threshold", "rgb_thr"}, 205);
  double hsv_v_thr = params.getd({"hsv_v_threshold", "hsv_v_thr"}, 190);
  double hsv_s_upper = params.getd("hsv_s_upper", 150);
  double oil_s_upper = params.getd("oil_shine_s_upper", std::max(165.0, hsv_s_upper));
  double lab_l_thr = params.getd({"lab_l_threshold", "lab_l_thr"}, 188);
  double sat_thr = params.getd({"saturation_pixel_threshold", "saturated_pixel_threshold"}, 245);
  double local_thr = params.getd({"local_brightness_threshold", "local_brightness_delta"}, 5);
  double local_ratio_thr = params.getd("local_contrast_threshold", 0.025);

  cv::Mat local_l = masked_gaussian(l, base, params.getd("local_sigma", 6.0));
  cv::Mat large_l = masked_gaussian(l, base, params.getd("large_local_sigma", 28.0));
  cv::Mat local_diff(h, w, CV_32F), large_diff(h, w, CV_32F), local_ratio(h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    const float* pl = l.ptr<float>(y);
    const float* pll = local_l.ptr<float>(y);
    const float* plgl = large_l.ptr<float>(y);
    float* pld = local_diff.ptr<float>(y);
    float* plg = large_diff.ptr<float>(y);
    float* plr = local_ratio.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      pld[x] = pl[x] - pll[x];
      plg[x] = pl[x] - plgl[x];
      plr[x] = pld[x] / std::max(pll[x], 1.0f);
    }
  }

  int win = params.geti("local_window_radius", 31);
  win = std::max(7, win % 2 == 1 ? win : win + 1);
  cv::Mat base_f(h, w, CV_32F), a_prod(h, w, CV_32F), b_prod(h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    float* bf = base_f.ptr<float>(y);
    float* ap = a_prod.ptr<float>(y);
    float* bp = b_prod.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      bf[x] = bm[x] ? 1.0f : 0.0f;
      ap[x] = pa[x] * bf[x];
      bp[x] = pb[x] * bf[x];
    }
  }
  cv::Mat w_sum, a_blur, b_blur;
  cv::GaussianBlur(base_f, w_sum, cv::Size(win, win), 0);
  cv::GaussianBlur(a_prod, a_blur, cv::Size(win, win), 0);
  cv::GaussianBlur(b_prod, b_blur, cv::Size(win, win), 0);
  cv::Mat chroma_delta(h, w, CV_32F);
  double chroma_upper = params.getd("local_skin_chroma_delta_upper", 42);
  for (int y = 0; y < h; ++y) {
    const float* ws = w_sum.ptr<float>(y);
    const float* ab = a_blur.ptr<float>(y);
    const float* bb = b_blur.ptr<float>(y);
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    float* cd = chroma_delta.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      float wsx = ws[x] + 1e-3f;
      float al = ab[x] / wsx;
      float bl = bb[x] / wsx;
      float da = pa[x] - al;
      float db = pb[x] - bl;
      cd[x] = std::sqrt(da * da + db * db);
    }
  }

  // 传统强证据规则。
  cv::Mat classic(h, w, CV_8U);
  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const float* prm = rgb_max.ptr<float>(y);
    const float* prr = rgb_range.ptr<float>(y);
    const float* pv = v.ptr<float>(y);
    const float* ps = s.ptr<float>(y);
    const float* pl = l.ptr<float>(y);
    const float* pld = local_diff.ptr<float>(y);
    const float* plg = large_diff.ptr<float>(y);
    const float* plr = local_ratio.ptr<float>(y);
    const float* pcd = chroma_delta.ptr<float>(y);
    uint8_t* pc = classic.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      bool rule_rgb = prm[x] >= static_cast<float>(rgb_thr) ||
                      (prm[x] >= static_cast<float>(sat_thr) && prr[x] <= 96);
      bool rule_hsv_white = pv[x] >= static_cast<float>(hsv_v_thr) && ps[x] <= static_cast<float>(hsv_s_upper);
      bool rule_hsv_oil = pv[x] >= static_cast<float>(hsv_v_thr - 8) && ps[x] <= static_cast<float>(oil_s_upper);
      bool rule_lab = pl[x] >= static_cast<float>(lab_l_thr);
      bool rule_local = pld[x] >= static_cast<float>(local_thr) ||
                        plr[x] >= static_cast<float>(local_ratio_thr) ||
                        plg[x] >= static_cast<float>(std::max(2.0, local_thr * 0.45));
      int score_count = (rule_rgb ? 1 : 0) + (rule_hsv_white ? 1 : 0) + (rule_hsv_oil ? 1 : 0) +
                        (rule_lab ? 1 : 0) + (rule_local ? 1 : 0);
      bool chroma_ok = pcd[x] <= static_cast<float>(chroma_upper);
      bool c = bm[x] && chroma_ok &&
               ((score_count >= 2 && (rule_lab || rule_hsv_oil || rule_local)) ||
                (score_count >= 1 && rule_local && (rule_rgb || rule_lab || rule_hsv_oil)));
      pc[x] = c ? 1 : 0;
    }
  }

  cv::Mat adaptive_core = cv::Mat::zeros(h, w, CV_8U);
  cv::Mat adaptive_grown = cv::Mat::zeros(h, w, CV_8U);
  cv::Mat global_score = cv::Mat::zeros(h, w, CV_32F);

  std::vector<std::string> region_keys;
  if (regions.masks.count("treatable_skin")) {
    region_keys = {"highlight_candidate"};
  } else {
    region_keys = {"forehead",        "nose_bridge",      "nose_tip",   "philtrum",
                   "left_brow_ridge", "right_brow_ridge", "left_cheek", "right_cheek",
                   "chin"};
  }
  for (const auto& key : region_keys) {
    auto it = regions.masks.find(key);
    if (it == regions.masks.end()) continue;
    AdaptiveResult r = adaptive_region_mask(key, it->second, base, rgb_max, rgb_range, gray, v, s,
                                            l, chroma_delta, local_diff, large_diff, params);
    for (int y = 0; y < h; ++y) {
      const uint8_t* rc = r.core.ptr<uint8_t>(y);
      const uint8_t* rg = r.grown.ptr<uint8_t>(y);
      const float* rs = r.score.ptr<float>(y);
      uint8_t* ac = adaptive_core.ptr<uint8_t>(y);
      uint8_t* ag = adaptive_grown.ptr<uint8_t>(y);
      float* gs = global_score.ptr<float>(y);
      for (int x = 0; x < w; ++x) {
        ac[x] |= rc[x];
        ag[x] |= rg[x];
        gs[x] = std::max(gs[x], rs[x]);
      }
    }
  }

  cv::Mat hard(h, w, CV_8U);
  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const uint8_t* pc = classic.ptr<uint8_t>(y);
    const uint8_t* ac = adaptive_core.ptr<uint8_t>(y);
    const uint8_t* ag = adaptive_grown.ptr<uint8_t>(y);
    uint8_t* ph = hard.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) ph[x] = (bm[x] && (pc[x] || ac[x] || ag[x])) ? 255 : 0;
  }

  int close_r = params.geti("morph_close_radius", 3);
  int dilate_r = params.geti({"mask_dilate_radius", "morph_dilate_radius"}, 1);
  int erode_r = params.geti({"mask_erode_radius", "morph_erode_radius"}, 0);
  int blur_r = params.geti("mask_blur_radius", 9);

  if (close_r > 0) {
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * close_r + 1, 2 * close_r + 1));
    cv::morphologyEx(hard, hard, cv::MORPH_CLOSE, k);
  }
  if (erode_r > 0) hard = morph(hard, "erode", erode_r);
  if (dilate_r > 0) hard = morph(hard, "dilate", dilate_r);

  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    uint8_t* ph = hard.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      if (!bm[x]) ph[x] = 0;
      if (pr[x] > 0) ph[x] = 0;
    }
  }
  hard = remove_small_components(hard, params.geti({"highlight_min_area", "min_area"}, 8));

  cv::Mat face_mask_u8 = regions.masks.count("face_mask") ? regions.masks.at("face_mask")
                         : regions.masks.count("skin")    ? regions.masks.at("skin")
                                                          : hard;
  long long face_area = std::max<long long>(1, cv::countNonZero(face_mask_u8 > 0));
  long long max_area = static_cast<long long>(
      face_area * params.getd({"highlight_max_area_ratio", "max_area_ratio"}, 0.125));
  long long hard_count = cv::countNonZero(hard);
  if (max_area > 0 && hard_count > max_area) {
    cv::Mat strength(h, w, CV_32F);
    for (int y = 0; y < h; ++y) {
      const float* gs = global_score.ptr<float>(y);
      const float* pl = l.ptr<float>(y);
      const float* pv = v.ptr<float>(y);
      const float* pld = local_diff.ptr<float>(y);
      float* st = strength.ptr<float>(y);
      for (int x = 0; x < w; ++x) {
        float t1 = std::clamp((pl[x] - static_cast<float>(lab_l_thr)) / 20.0f, 0.0f, 3.0f);
        float t2 = std::clamp((pv[x] - static_cast<float>(hsv_v_thr)) / 25.0f, 0.0f, 3.0f);
        float t3 = std::clamp(pld[x] / static_cast<float>(std::max(local_thr, 1.0)), 0.0f, 3.0f);
        st[x] = gs[x] + 0.18f * t1 + 0.12f * t2 + 0.10f * t3;
      }
    }
    cv::Mat hard_bool = hard > 0;
    cv::Mat hard_mask01(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
      const uint8_t* hb = hard_bool.ptr<uint8_t>(y);
      uint8_t* o = hard_mask01.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) o[x] = hb[x] ? 1 : 0;
    }
    std::vector<float> vals = masked_values(strength, hard_mask01);
    if (static_cast<long long>(vals.size()) > max_area) {
      std::nth_element(vals.begin(), vals.end() - max_area, vals.end());
      float kth = vals[vals.size() - static_cast<size_t>(max_area)];
      for (int y = 0; y < h; ++y) {
        const uint8_t* bm = base.ptr<uint8_t>(y);
        const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
        const float* st = strength.ptr<float>(y);
        uint8_t* ph = hard.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
          ph[x] = (ph[x] > 0 && st[x] >= kth && bm[x]) ? 255 : 0;
          if (pr[x] > 0) ph[x] = 0;
        }
      }
    }
  }

  if (cv::countNonZero(hard) == 0)
    out.warnings.push_back("未检测到满足条件的高光区域，可降低 HSV V、Lab L、局部亮度异常阈值或提高油光饱和度上限");

  bool overlap = false;
  for (int y = 0; y < h && !overlap; ++y) {
    const uint8_t* ph = hard.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x)
      if (ph[x] > 0 && pr[x] > 0) {
        overlap = true;
        break;
      }
  }
  if (overlap) {
    out.warnings.push_back("眼睛或嘴唇区域被误选中，已自动从高光 mask 中排除");
    for (int y = 0; y < h; ++y) {
      const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
      uint8_t* ph = hard.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x)
        if (pr[x] > 0) ph[x] = 0;
    }
  }

  cv::Mat soft = blur_r > 0 ? blur_mask(hard, blur_r) : hard.clone();
  double soft_gain = params.getd("soft_mask_gain", 1.65);
  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const uint8_t* pr = protect_u8.ptr<uint8_t>(y);
    uint8_t* po = soft.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      // numpy：clip(soft*gain, 0, 255).astype(uint8) → 截断取整。
      float val = static_cast<float>(po[x]) * static_cast<float>(soft_gain);
      val = std::clamp(val, 0.0f, 255.0f);
      uint8_t vv = static_cast<uint8_t>(val);
      if (!bm[x] || pr[x] > 0) vv = 0;
      po[x] = vv;
    }
  }

  out.hard_mask = hard;
  out.soft_mask = soft;
  return out;
}

}  // namespace facehi
