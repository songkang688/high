// 对应 highlight_removal/skin_mask.py。
#include <opencv2/imgproc.hpp>

#include "facehi/face_regions.hpp"
#include "facehi/utils.hpp"

namespace facehi {

cv::Mat estimate_skin_mask(const cv::Mat& image_bgr, const cv::Mat& face_mask,
                           const cv::Mat& protect_mask, const Params& params) {
  const int h = image_bgr.rows, w = image_bgr.cols;
  cv::Mat base(h, w, CV_8U);
  int base_count = 0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* f = face_mask.ptr<uint8_t>(y);
    const uint8_t* p = protect_mask.ptr<uint8_t>(y);
    uint8_t* b = base.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      b[x] = (f[x] > 0 && p[x] == 0) ? 1 : 0;
      base_count += b[x];
    }
  }
  if (base_count == 0) return cv::Mat::zeros(h, w, CV_8U);

  cv::Mat hsv, ycrcb, lab;
  cv::cvtColor(image_bgr, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(image_bgr, ycrcb, cv::COLOR_BGR2YCrCb);
  cv::cvtColor(image_bgr, lab, cv::COLOR_BGR2Lab);

  // sample = base & (v>40) & (v<245) & (s>8)；不足 50 像素则回退 base & (v>25)。
  std::vector<float> cr_s, cb_s, a_s, b_s;
  auto collect = [&](bool relaxed) {
    cr_s.clear();
    cb_s.clear();
    a_s.clear();
    b_s.clear();
    for (int y = 0; y < h; ++y) {
      const uint8_t* bm = base.ptr<uint8_t>(y);
      const cv::Vec3b* hv = hsv.ptr<cv::Vec3b>(y);
      const cv::Vec3b* yc = ycrcb.ptr<cv::Vec3b>(y);
      const cv::Vec3b* lb = lab.ptr<cv::Vec3b>(y);
      for (int x = 0; x < w; ++x) {
        if (!bm[x]) continue;
        float v = hv[x][2], s = hv[x][1];
        bool ok = relaxed ? (v > 25) : (v > 40 && v < 245 && s > 8);
        if (!ok) continue;
        cr_s.push_back(yc[x][1]);
        cb_s.push_back(yc[x][2]);
        a_s.push_back(lb[x][1]);
        b_s.push_back(lb[x][2]);
      }
    }
  };
  collect(false);
  if (cr_s.size() < 50) collect(true);

  double med_cr = np_median(cr_s, 150);
  double med_cb = np_median(cb_s, 110);
  double med_a = np_median(a_s, 135);
  double med_b = np_median(b_s, 135);

  cv::Mat skin_u8(h, w, CV_8U);
  for (int y = 0; y < h; ++y) {
    const uint8_t* bm = base.ptr<uint8_t>(y);
    const cv::Vec3b* hv = hsv.ptr<cv::Vec3b>(y);
    const cv::Vec3b* yc = ycrcb.ptr<cv::Vec3b>(y);
    const cv::Vec3b* lb = lab.ptr<cv::Vec3b>(y);
    uint8_t* o = skin_u8.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      float v = hv[x][2], s = hv[x][1];
      float cr = yc[x][1], cb = yc[x][2];
      float a = lb[x][1], b = lb[x][2];
      bool generic = cr >= 125 && cr <= 185 && cb >= 70 && cb <= 145 && v >= 35;
      // numpy 中 np.median 返回 float64 标量（强类型），比较在双精度下进行。
      bool adaptive = std::abs(static_cast<double>(cr) - med_cr) <= 34 &&
                      std::abs(static_cast<double>(cb) - med_cb) <= 34 &&
                      std::abs(static_cast<double>(a) - med_a) <= 28 &&
                      std::abs(static_cast<double>(b) - med_b) <= 34 && v >= 35;
      bool bright_highlight_like = v >= 165 && s <= 175;
      bool dark_non_skin = v < 45 || (v < 85 && s > 70);
      bool skin = bm[x] && (generic || adaptive || bright_highlight_like) && !dark_non_skin;
      o[x] = skin ? 255 : 0;
    }
  }

  int smooth_radius = params.geti("skin_mask_smooth_radius", 3);
  skin_u8 = morph(skin_u8, "close", std::max(1, smooth_radius));
  skin_u8 = morph(skin_u8, "open", 1);
  if (smooth_radius > 0) {
    skin_u8 = blur_mask(skin_u8, smooth_radius);
    for (int y = 0; y < h; ++y) {
      uint8_t* o = skin_u8.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) o[x] = o[x] > 80 ? 255 : 0;
    }
  }
  for (int y = 0; y < h; ++y) {
    const uint8_t* p = protect_mask.ptr<uint8_t>(y);
    uint8_t* o = skin_u8.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x)
      if (p[x] > 0) o[x] = 0;
  }
  return skin_u8;
}

}  // namespace facehi
