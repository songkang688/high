// 对应 highlight_removal/face_regions.py（simple_protect_mode 与完整模式）。
#include "facehi/face_regions.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "facehi/utils.hpp"

namespace facehi {

namespace {

// ---------------------------------------------------------------------------
// numpy.polyfit(deg=2) 兼容实现：列缩放 + 最小二乘（双精度正规方程）。
// ---------------------------------------------------------------------------
std::array<double, 3> polyfit2(const std::vector<double>& x, const std::vector<double>& y) {
  const size_t n = x.size();
  // Vandermonde 列 [x^2, x, 1]，先按 numpy 方式做列 L2 归一化。
  std::array<double, 3> scale{0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    double c0 = x[i] * x[i], c1 = x[i], c2 = 1.0;
    scale[0] += c0 * c0;
    scale[1] += c1 * c1;
    scale[2] += c2 * c2;
  }
  for (auto& s : scale) s = std::sqrt(s);
  // 正规方程 A^T A c = A^T y（缩放后条件数良好）。
  double ata[3][3] = {{0}};
  double aty[3] = {0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    double col[3] = {x[i] * x[i] / scale[0], x[i] / scale[1], 1.0 / scale[2]};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) ata[r][c] += col[r] * col[c];
      aty[r] += col[r] * y[i];
    }
  }
  // 3x3 高斯消元（部分选主元）。
  double m[3][4];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[r][c] = ata[r][c];
    m[r][3] = aty[r];
  }
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c + 1; r < 3; ++r)
      if (std::abs(m[r][c]) > std::abs(m[piv][c])) piv = r;
    if (piv != c)
      for (int k = 0; k < 4; ++k) std::swap(m[c][k], m[piv][k]);
    for (int r = c + 1; r < 3; ++r) {
      double f = m[r][c] / m[c][c];
      for (int k = c; k < 4; ++k) m[r][k] -= f * m[c][k];
    }
  }
  std::array<double, 3> sol{0, 0, 0};
  for (int r = 2; r >= 0; --r) {
    double s = m[r][3];
    for (int c = r + 1; c < 3; ++c) s -= m[r][c] * sol[c];
    sol[r] = s / m[r][r];
  }
  for (int i = 0; i < 3; ++i) sol[i] /= scale[i];
  return sol;  // c0*x^2 + c1*x + c2
}

// np.interp：xp 升序，越界取端点。
float np_interp(float xq, const std::vector<float>& xp, const std::vector<float>& fp) {
  if (xq <= xp.front()) return fp.front();
  if (xq >= xp.back()) return fp.back();
  size_t lo = 0, hi = xp.size() - 1;
  while (hi - lo > 1) {
    size_t mid = (lo + hi) / 2;
    if (xp[mid] <= xq)
      lo = mid;
    else
      hi = mid;
  }
  double slope = (static_cast<double>(fp[hi]) - fp[lo]) / (static_cast<double>(xp[hi]) - xp[lo]);
  return static_cast<float>(fp[lo] + slope * (static_cast<double>(xq) - xp[lo]));
}

// 一维行向量高斯平滑：等价 cv2.GaussianBlur(y.reshape(1,-1), (k,1), 0)。
void row_gaussian(std::vector<float>* y, int k) {
  cv::Mat row(1, static_cast<int>(y->size()), CV_32F, y->data());
  cv::Mat out;
  cv::GaussianBlur(row, out, cv::Size(k, 1), 0);
  out.copyTo(row);
}

cv::Mat protect_feature(const cv::Size& shape, const cv::Mat& landmarks,
                        const std::vector<int>& indices, int dilate_px) {
  cv::Mat mask = fill_hull_mask(shape, lm_pts(landmarks, indices));
  if (dilate_px > 0) mask = morph(mask, "dilate", dilate_px);
  return mask;
}

struct PhiltrumBox {
  int y0, y1, x0, x1;
};

std::optional<PhiltrumBox> philtrum_box(const cv::Size& shape, const cv::Mat& landmarks,
                                        double io, double width_ratio) {
  const int h = shape.height, w = shape.width;
  auto upper_lip = lm_pts(landmarks, {61, 0, 37, 39, 40, 185, 291, 375, 321, 405, 314, 17, 267, 269, 270});
  auto nose_bottom = lm_pts(landmarks, {2, 94, 19, 4, 5, 98, 327, 326, 49, 279});
  if (upper_lip.size() < 2 || nose_bottom.size() < 2) return std::nullopt;
  double lip_top_y = upper_lip[0].y, nose_base_y = nose_bottom[0].y;
  for (const auto& p : upper_lip) lip_top_y = std::min(lip_top_y, static_cast<double>(p.y));
  for (const auto& p : nose_bottom) nose_base_y = std::max(nose_base_y, static_cast<double>(p.y));
  if (lip_top_y <= nose_base_y + 1) return std::nullopt;
  double sx = 0;
  for (const auto& p : upper_lip) sx += p.x;
  double cx = static_cast<double>(static_cast<float>(sx / upper_lip.size()));
  double half_w = std::max(2.0, io * width_ratio);
  int y0 = static_cast<int>(std::max(0.0, std::floor(nose_base_y - io * 0.02)));
  int y1 = static_cast<int>(std::min(static_cast<double>(h), std::ceil(lip_top_y + io * 0.04)));
  int x0 = static_cast<int>(std::max(0.0, std::floor(cx - half_w)));
  int x1 = static_cast<int>(std::min(static_cast<double>(w), std::ceil(cx + half_w)));
  if (y1 <= y0 || x1 <= x0) return std::nullopt;
  return PhiltrumBox{y0, y1, x0, x1};
}

cv::Mat carve_philtrum_from_protect(const cv::Mat& protect, const cv::Mat& landmarks,
                                    double io, double width_ratio) {
  auto box = philtrum_box(protect.size(), landmarks, io, width_ratio);
  if (!box) return protect;
  cv::Mat out = protect.clone();
  out(cv::Rect(box->x0, box->y0, box->x1 - box->x0, box->y1 - box->y0)).setTo(0);
  return out;
}

cv::Mat build_philtrum_mask(const cv::Size& shape, const cv::Mat& landmarks, double io,
                            const Params& params) {
  double width_ratio = params.getd("philtrum_width_ratio", 0.16);
  auto box = philtrum_box(shape, landmarks, io, width_ratio * 1.08);
  cv::Mat mask = cv::Mat::zeros(shape, CV_8U);
  if (!box) return mask;
  mask(cv::Rect(box->x0, box->y0, box->x1 - box->x0, box->y1 - box->y0)).setTo(255);
  return mask;
}

// 对应 _smooth_brow_arc_curve：返回 (y 曲线 float32, x_left, x_right)。
void smooth_brow_arc_curve(const cv::Size& shape, std::vector<cv::Point2f> brow_pts, double io,
                           double width_ratio, double y_shift_ratio,
                           std::vector<float>* y_out, double* x_left, double* x_right) {
  const int h = shape.height, w = shape.width;
  y_out->assign(w, static_cast<float>(h));
  if (brow_pts.size() < 2) {
    *x_left = 0.0;
    *x_right = static_cast<double>(w - 1);
    return;
  }
  std::stable_sort(brow_pts.begin(), brow_pts.end(),
                   [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
  std::vector<double> bx, by;
  for (const auto& p : brow_pts) {
    bx.push_back(p.x);
    by.push_back(p.y);
  }
  std::vector<float>& y = *y_out;
  if (bx.size() >= 3) {
    auto c = polyfit2(bx, by);
    for (int i = 0; i < w; ++i) {
      double xv = static_cast<double>(static_cast<float>(i));
      y[i] = static_cast<float>((c[0] * xv + c[1]) * xv + c[2]);
    }
  } else {
    std::vector<float> xp, fp;
    for (size_t i = 0; i < bx.size(); ++i) {
      xp.push_back(static_cast<float>(bx[i]));
      fp.push_back(static_cast<float>(by[i]));
    }
    for (int i = 0; i < w; ++i) y[i] = np_interp(static_cast<float>(i), xp, fp);
  }
  long long k = std::max<long long>(11, pyround(io * 0.34));
  if (k % 2 == 0) k += 1;
  row_gaussian(&y, static_cast<int>(k));
  float shift = static_cast<float>(io * std::max(0.0, y_shift_ratio));
  for (auto& v : y) v -= shift;
  double ext = io * 0.028 * std::max(0.5, width_ratio);
  *x_left = bx.front() - ext;
  *x_right = bx.back() + ext;
}

cv::Mat arc_band_mask(const cv::Size& shape, const std::vector<float>& y_top,
                      const std::vector<float>& y_bot, double x_left, double x_right) {
  const int h = shape.height, w = shape.width;
  cv::Mat out = cv::Mat::zeros(shape, CV_8U);
  float xl = static_cast<float>(x_left), xr = static_cast<float>(x_right);
  for (int yv = 0; yv < h; ++yv) {
    uint8_t* o = out.ptr<uint8_t>(yv);
    float yf = static_cast<float>(yv);
    for (int xv = 0; xv < w; ++xv) {
      float xf = static_cast<float>(xv);
      if (xf >= xl && xf <= xr && yf >= y_top[xv] && yf <= y_bot[xv]) o[xv] = 255;
    }
  }
  return out;
}

void brow_protect_thickness(double io, int dilate_px, int shrink_px, double height_ratio,
                            double* pad_up, double* pad_down) {
  double hr = std::max(0.5, height_ratio);
  *pad_up = io * 0.016 * hr;
  *pad_down = io * 0.009 * hr;
  *pad_down += std::max(0, dilate_px - shrink_px) * io * 0.004 * hr;
}

cv::Mat build_brow_protect_mask(const cv::Size& shape, const cv::Mat& landmarks,
                                const std::vector<int>& brow_indices,
                                const std::vector<int>& eye_indices, double io, int dilate_px,
                                int shrink_px, double height_ratio, double width_ratio,
                                double y_shift_ratio) {
  auto eye_pts = lm_pts(landmarks, eye_indices);
  auto brow = filter_side_brow_pts(lm_pts(landmarks, brow_indices), eye_pts, io);
  if (brow.size() < 2 || eye_pts.size() < 2) return cv::Mat::zeros(shape, CV_8U);
  float ex0 = eye_pts[0].x, ex1 = eye_pts[0].x;
  for (const auto& p : eye_pts) {
    ex0 = std::min(ex0, p.x);
    ex1 = std::max(ex1, p.x);
  }
  if (static_cast<double>(ex1 - ex0) < io * 0.07) return cv::Mat::zeros(shape, CV_8U);
  std::vector<float> y_c;
  double x_l, x_r;
  smooth_brow_arc_curve(shape, brow, io, width_ratio, y_shift_ratio, &y_c, &x_l, &x_r);
  double pad_up, pad_down;
  brow_protect_thickness(io, dilate_px, shrink_px, height_ratio, &pad_up, &pad_down);
  std::vector<float> y_top(y_c.size()), y_bot(y_c.size());
  for (size_t i = 0; i < y_c.size(); ++i) {
    y_top[i] = y_c[i] - static_cast<float>(pad_up);
    y_bot[i] = y_c[i] + static_cast<float>(pad_down);
  }
  return arc_band_mask(shape, y_top, y_bot, x_l, x_r);
}

// 对应 _side_brow_top_1d。
std::vector<float> side_brow_top_1d(const cv::Size& shape, const std::vector<cv::Point2f>& brow_pts,
                                    double io, double width_ratio, double y_shift_ratio,
                                    double pad_up) {
  const int h = shape.height, w = shape.width;
  std::vector<float> y_out(w, static_cast<float>(h));
  if (brow_pts.size() < 2) return y_out;
  std::vector<float> y_c;
  double x_l, x_r;
  smooth_brow_arc_curve(shape, brow_pts, io, width_ratio, y_shift_ratio, &y_c, &x_l, &x_r);
  float xl = static_cast<float>(x_l), xr = static_cast<float>(x_r);
  float off = static_cast<float>(pad_up) + static_cast<float>(io * 0.003);
  for (int i = 0; i < w; ++i) {
    float xf = static_cast<float>(i);
    if (xf >= xl && xf <= xr) y_out[i] = y_c[i] - static_cast<float>(pad_up) - static_cast<float>(io * 0.003);
  }
  (void)off;
  return y_out;
}

std::vector<float> forehead_brow_cutoff_y_map(const cv::Size& shape, const cv::Mat& landmarks,
                                              double io, double y_shift_ratio, double height_ratio,
                                              int dilate_px, int shrink_px) {
  const int h = shape.height, w = shape.width;
  double pad_up, pad_down;
  brow_protect_thickness(io, dilate_px, shrink_px, height_ratio, &pad_up, &pad_down);
  std::vector<float> y_cut(w, static_cast<float>(h));
  const std::vector<std::pair<const std::vector<int>*, const std::vector<int>*>> sides = {
      {&LEFT_BROW, &LEFT_EYE}, {&RIGHT_BROW, &RIGHT_EYE}};
  for (const auto& [brow_ids, eye_ids] : sides) {
    auto brow = filter_side_brow_pts(lm_pts(landmarks, *brow_ids), lm_pts(landmarks, *eye_ids), io);
    auto top = side_brow_top_1d(shape, brow, io, 1.0, y_shift_ratio, pad_up);
    for (int i = 0; i < w; ++i) y_cut[i] = std::min(y_cut[i], top[i]);
  }
  return y_cut;
}

double compute_forehead_lift_px(const cv::Mat& landmarks, double io,
                                const std::vector<cv::Point2f>& oval_pts, double fh_expand) {
  auto lb = lm_pts(landmarks, LEFT_BROW);
  auto rb = lm_pts(landmarks, RIGHT_BROW);
  std::vector<cv::Point2f> brow_pts = lb;
  brow_pts.insert(brow_pts.end(), rb.begin(), rb.end());
  if (brow_pts.size() < 2 || oval_pts.size() < 3) return io * 0.20;
  std::vector<float> brow_y;
  for (const auto& p : brow_pts) brow_y.push_back(p.y);
  double brow_bottom_y = np_percentile(brow_y, 72.0);
  float oval_min_y = oval_pts[0].y;
  for (const auto& p : oval_pts) oval_min_y = std::min(oval_min_y, p.y);
  double span = std::max({brow_bottom_y - static_cast<double>(oval_min_y), io * 0.20, 4.0});
  if (fh_expand <= 1.0) return io * 0.20 * std::max(0.35, fh_expand);
  return io * 0.20 + (fh_expand - 1.0) * span;
}

std::vector<cv::Point2f> lift_face_oval_pts(const cv::Mat& landmarks, double io,
                                            double forehead_height_ratio) {
  auto oval_pts = lm_pts(landmarks, FACE_OVAL);
  if (oval_pts.size() < 3) return oval_pts;
  auto lb = lm_pts(landmarks, LEFT_BROW);
  auto rb = lm_pts(landmarks, RIGHT_BROW);
  std::vector<cv::Point2f> brow_pts = lb;
  brow_pts.insert(brow_pts.end(), rb.begin(), rb.end());
  double brow_bottom_y;
  if (brow_pts.size() >= 2) {
    std::vector<float> brow_y;
    for (const auto& p : brow_pts) brow_y.push_back(p.y);
    brow_bottom_y = np_percentile(brow_y, 72.0);
  } else {
    float oval_min_y = oval_pts[0].y;
    for (const auto& p : oval_pts) oval_min_y = std::min(oval_min_y, p.y);
    brow_bottom_y = static_cast<double>(oval_min_y) + io * 0.35;
  }
  double lift_px = compute_forehead_lift_px(landmarks, io, oval_pts, forehead_height_ratio);
  if (lift_px <= 0.5) return oval_pts;
  auto lifted = oval_pts;
  float thr = static_cast<float>(brow_bottom_y - 1.0);
  float lift_f = static_cast<float>(lift_px);
  for (auto& p : lifted)
    if (p.y < thr) p.y = std::max(0.0f, p.y - lift_f);
  return lifted;
}

cv::Mat forehead_support_oval(const cv::Size& shape, const cv::Mat& landmarks,
                              const cv::Mat& face_mask_raw, double lift_px, double brow_bottom_y) {
  if (lift_px <= 0.5) return face_mask_raw;
  auto oval_pts = lm_pts(landmarks, FACE_OVAL);
  if (oval_pts.size() < 3) return face_mask_raw;
  auto lifted = oval_pts;
  float thr = static_cast<float>(brow_bottom_y - 1.0);
  float lift_f = static_cast<float>(lift_px);
  for (auto& p : lifted)
    if (p.y < thr) p.y = std::max(0.0f, p.y - lift_f);
  cv::Mat out = fill_poly_mask(shape, lifted);
  if (cv::countNonZero(out) == 0) out = fill_hull_mask(shape, lifted);
  return cv::countNonZero(out) ? out : face_mask_raw;
}

void build_lifted_face_mask_raw(const cv::Size& shape, const cv::Mat& landmarks, double io,
                                const Params& params, cv::Mat* face_mask_raw,
                                std::vector<cv::Point2f>* contour_pts) {
  double fh_expand = params.getd("forehead_height_ratio", 1.28);
  *contour_pts = lift_face_oval_pts(landmarks, io, fh_expand);
  *face_mask_raw = fill_poly_mask(shape, *contour_pts);
  if (cv::countNonZero(*face_mask_raw) == 0) *face_mask_raw = fill_hull_mask(shape, *contour_pts);
  if (cv::countNonZero(*face_mask_raw) == 0) {
    *contour_pts = lm_pts(landmarks, FACE_OVAL);
    *face_mask_raw = fill_poly_mask(shape, *contour_pts);
    if (cv::countNonZero(*face_mask_raw) == 0) *face_mask_raw = fill_hull_mask(shape, *contour_pts);
  }
}

cv::Mat build_forehead_mask(const cv::Size& shape, const cv::Mat& landmarks, double io,
                            const cv::Mat& face_mask_raw, const Params& params) {
  double fh_expand = params.getd("forehead_height_ratio", 1.0);
  auto lb = lm_pts(landmarks, LEFT_BROW);
  auto rb = lm_pts(landmarks, RIGHT_BROW);
  std::vector<cv::Point2f> brow_pts = lb;
  brow_pts.insert(brow_pts.end(), rb.begin(), rb.end());
  if (brow_pts.size() < 2) return cv::Mat::zeros(shape, CV_8U);

  std::vector<float> brow_y;
  for (const auto& p : brow_pts) brow_y.push_back(p.y);
  double brow_bottom_y = np_percentile(brow_y, 72.0);
  double lift_px = compute_forehead_lift_px(landmarks, io, lm_pts(landmarks, FACE_OVAL), fh_expand);
  cv::Mat oval = forehead_support_oval(shape, landmarks, face_mask_raw, lift_px, brow_bottom_y);

  const int h = shape.height, w = shape.width;
  int brow_shrink = params.geti("brow_protect_shrink_radius", 2);
  int protect_radius = params.geti("protect_expand_radius", 6);
  double y_shift = params.getd("brow_vertical_shift_ratio", 0.0);
  double height_ratio = params.getd("brow_protect_height_ratio", 1.0);
  auto y_cut = forehead_brow_cutoff_y_map(shape, landmarks, io, y_shift, height_ratio,
                                          protect_radius, brow_shrink);
  cv::Mat forehead = cv::Mat::zeros(shape, CV_8U);
  for (int y = 0; y < h; ++y) {
    const uint8_t* ov = oval.ptr<uint8_t>(y);
    uint8_t* o = forehead.ptr<uint8_t>(y);
    float yf = static_cast<float>(y);
    for (int x = 0; x < w; ++x)
      if (ov[x] > 0 && yf < y_cut[x]) o[x] = 255;
  }

  if (fh_expand < 1.0 && cv::countNonZero(forehead) > 0) {
    int min_y = h, max_y = -1;
    for (int y = 0; y < h; ++y) {
      const uint8_t* o = forehead.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x)
        if (o[x] > 0) {
          min_y = std::min(min_y, y);
          max_y = std::max(max_y, y);
          break;
        }
    }
    int span = std::max(4, max_y - min_y);
    int keep_from = max_y - static_cast<int>(span * std::max(0.35, fh_expand));
    for (int y = 0; y < std::min(h, keep_from); ++y) forehead.row(y).setTo(0);
  }
  return forehead;
}

cv::Mat clip_forehead(const cv::Mat& mask, const cv::Mat& protect) {
  cv::Mat out(mask.size(), CV_8U);
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* m = mask.ptr<uint8_t>(y);
    const uint8_t* p = protect.ptr<uint8_t>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) o[x] = (m[x] > 0 && p[x] == 0) ? 255 : 0;
  }
  return out;
}

cv::Mat clip_to_skin(const cv::Mat& mask, const cv::Mat& skin, const cv::Mat& protect) {
  cv::Mat out(mask.size(), CV_8U);
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* m = mask.ptr<uint8_t>(y);
    const uint8_t* s = skin.ptr<uint8_t>(y);
    const uint8_t* p = protect.ptr<uint8_t>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) o[x] = (m[x] > 0 && s[x] > 0 && p[x] == 0) ? 255 : 0;
  }
  return out;
}

// 对应 _build_residual_brow_ridge_side。side_u_mask 用函数式判断。
cv::Mat build_residual_brow_ridge_side(
    const cv::Size& shape, std::vector<cv::Point2f> brow_pts, std::vector<cv::Point2f> eye_pts,
    double io, const cv::Mat& face_mask, const cv::Mat& skin, const cv::Mat& occupied,
    const std::function<bool(int, int)>& side_u_mask, double height_ratio, double width_ratio,
    int dilate_px, int shrink_px, double y_shift_ratio) {
  brow_pts = filter_side_brow_pts(brow_pts, eye_pts, io);
  if (brow_pts.size() < 2 || eye_pts.size() < 2) return cv::Mat::zeros(shape, CV_8U);
  float ex_min = eye_pts[0].x, ex_max = eye_pts[0].x;
  for (const auto& p : eye_pts) {
    ex_min = std::min(ex_min, p.x);
    ex_max = std::max(ex_max, p.x);
  }
  if (static_cast<double>(ex_max - ex_min) < io * 0.07) return cv::Mat::zeros(shape, CV_8U);
  const int h = shape.height, w = shape.width;
  std::vector<float> y_c;
  double x_l, x_r;
  smooth_brow_arc_curve(shape, brow_pts, io, width_ratio, y_shift_ratio, &y_c, &x_l, &x_r);
  double pad_up, pad_down;
  brow_protect_thickness(io, dilate_px, shrink_px, height_ratio, &pad_up, &pad_down);
  std::vector<float> y_top(y_c.size());
  for (size_t i = 0; i < y_c.size(); ++i) y_top[i] = y_c[i] + static_cast<float>(pad_down);

  std::stable_sort(eye_pts.begin(), eye_pts.end(),
                   [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
  std::vector<float> ex, ey;
  for (const auto& p : eye_pts) {
    ex.push_back(p.x);
    ey.push_back(p.y);
  }
  std::vector<float> y_eye(w);
  for (int i = 0; i < w; ++i) y_eye[i] = np_interp(static_cast<float>(i), ex, ey);
  long long k = std::max<long long>(9, pyround(io * 0.18));
  if (k % 2 == 0) k += 1;
  row_gaussian(&y_eye, static_cast<int>(k));
  std::vector<float> y_bot(w);
  for (int i = 0; i < w; ++i) y_bot[i] = y_eye[i] + static_cast<float>(io * 0.018);

  cv::Mat out = cv::Mat::zeros(shape, CV_8U);
  float xl = static_cast<float>(x_l), xr = static_cast<float>(x_r);
  for (int y = 0; y < h; ++y) {
    const uint8_t* fm = face_mask.ptr<uint8_t>(y);
    const uint8_t* sk = skin.ptr<uint8_t>(y);
    const uint8_t* oc = occupied.ptr<uint8_t>(y);
    uint8_t* o = out.ptr<uint8_t>(y);
    float yf = static_cast<float>(y);
    for (int x = 0; x < w; ++x) {
      float xf = static_cast<float>(x);
      bool candidate = fm[x] > 0 && sk[x] > 0 && side_u_mask(x, y) && xf >= xl && xf <= xr &&
                       yf >= y_top[x] && yf <= y_bot[x];
      if (candidate && !(oc[x] > 0)) o[x] = 255;
    }
  }
  return out;
}

// simple 模式（默认）：只保护眉/眼/嘴。
RegionMasks build_face_regions_simple(const cv::Mat& image_bgr, const cv::Mat& landmarks,
                                      const Params& params) {
  const cv::Size shape(image_bgr.cols, image_bgr.rows);
  RegionMasks rm;
  rm.geometry = get_geometry(landmarks, shape);
  double io = std::max(1.0, rm.geometry.interocular);
  double head_yaw = estimate_head_yaw_ratio(landmarks);

  cv::Mat face_mask_raw;
  build_lifted_face_mask_raw(shape, landmarks, io, params, &face_mask_raw, &rm.face_contour_pts);
  int shrink_px = std::max<long long>(0, pyround(io * params.getd("face_contour_shrink_ratio", 0.02)));
  cv::Mat face_mask = shrink_px > 0 ? morph(face_mask_raw, "erode", shrink_px) : face_mask_raw.clone();

  int protect_radius = params.geti("protect_expand_radius", 6);
  double eye_extra = params.getd("eye_protect_extra_radius", 0.5);
  int brow_shrink = params.geti("brow_protect_shrink_radius", 2);
  double brow_height_ratio = params.getd("brow_protect_height_ratio", 1.0);
  double brow_width_ratio = params.getd("brow_protect_width_ratio", 1.0);
  double brow_y_shift = params.getd("brow_vertical_shift_ratio", 0.12);
  int lip_shrink = params.geti("lip_protect_shrink_radius", 2);
  int eye_radius = std::max<long long>(1, protect_radius + pyround(eye_extra));
  int lip_radius = std::max(1, protect_radius - lip_shrink);

  cv::Mat left_eye = protect_feature(shape, landmarks, LEFT_EYE, eye_radius);
  cv::Mat right_eye = protect_feature(shape, landmarks, RIGHT_EYE, eye_radius);
  std::vector<int> all_lips = OUTER_LIPS;
  all_lips.insert(all_lips.end(), INNER_LIPS.begin(), INNER_LIPS.end());
  cv::Mat lips = protect_feature(shape, landmarks, all_lips, lip_radius);
  cv::Mat left_brow_protect = build_brow_protect_mask(
      shape, landmarks, LEFT_BROW, LEFT_EYE, io, protect_radius, brow_shrink, brow_height_ratio,
      brow_width_ratio, brow_y_shift);
  cv::Mat right_brow_protect = build_brow_protect_mask(
      shape, landmarks, RIGHT_BROW, RIGHT_EYE, io, protect_radius, brow_shrink, brow_height_ratio,
      brow_width_ratio, brow_y_shift);
  cv::Mat protect;
  cv::bitwise_or(left_eye, right_eye, protect);
  cv::bitwise_or(protect, lips, protect);
  cv::bitwise_or(protect, left_brow_protect, protect);
  cv::bitwise_or(protect, right_brow_protect, protect);
  cv::bitwise_and(protect, face_mask_raw, protect);

  Params skin_params;
  skin_params.set("skin_mask_smooth_radius", params.getd("skin_mask_smooth_radius", 3));
  cv::Mat skin = estimate_skin_mask(image_bgr, face_mask, protect, skin_params);

  cv::Mat treatable(shape, CV_8U);
  for (int y = 0; y < shape.height; ++y) {
    const uint8_t* fm = face_mask.ptr<uint8_t>(y);
    const uint8_t* sk = skin.ptr<uint8_t>(y);
    const uint8_t* pr = protect.ptr<uint8_t>(y);
    uint8_t* t = treatable.ptr<uint8_t>(y);
    for (int x = 0; x < shape.width; ++x) t[x] = (fm[x] > 0 && sk[x] > 0 && pr[x] == 0) ? 255 : 0;
  }

  cv::Mat empty = cv::Mat::zeros(shape, CV_8U);
  if (cv::countNonZero(skin) == 0)
    rm.warnings.push_back("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点");
  if (cv::countNonZero(treatable) == 0)
    rm.warnings.push_back("可去高光皮肤区域为空，请检查人脸关键点");
  if (std::abs(head_yaw) > 0.18) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "侧脸角度较大（yaw≈%.2f），建议正脸或开启「人脸轻量对齐」", head_yaw);
    rm.warnings.push_back(buf);
  }

  rm.masks["face_mask_raw"] = face_mask_raw;
  rm.masks["face_mask"] = face_mask;
  rm.masks["skin"] = skin;
  rm.masks["protect"] = protect;
  rm.masks["left_eye"] = left_eye;
  rm.masks["right_eye"] = right_eye;
  rm.masks["left_brow_protect"] = left_brow_protect;
  rm.masks["right_brow_protect"] = right_brow_protect;
  rm.masks["left_brow_ridge"] = empty;
  rm.masks["right_brow_ridge"] = empty;
  rm.masks["lips"] = lips;
  rm.masks["mouth"] = lips;
  rm.masks["nose_bridge"] = empty;
  rm.masks["nose_tip"] = empty;
  rm.masks["forehead"] = empty;
  rm.masks["left_cheek"] = empty;
  rm.masks["right_cheek"] = empty;
  rm.masks["chin"] = empty;
  rm.masks["philtrum"] = empty;
  rm.masks["highlight_candidate"] = treatable;
  rm.masks["treatable_skin"] = treatable;
  return rm;
}

bool use_simple_protect_mode(const Params& params) {
  if (params.getb("key_region_detection_mode", false)) return false;
  return params.getb("simple_protect_mode", true);
}

}  // namespace

RegionMasks build_face_regions(const cv::Mat& image_bgr, const cv::Mat& landmarks,
                               const Params& params) {
  if (use_simple_protect_mode(params)) return build_face_regions_simple(image_bgr, landmarks, params);

  const cv::Size shape(image_bgr.cols, image_bgr.rows);
  const int h = shape.height, w = shape.width;
  RegionMasks rm;
  rm.geometry = get_geometry(landmarks, shape);
  const FaceGeometry& geo = rm.geometry;
  double io = std::max(1.0, geo.interocular);

  cv::Mat face_mask_raw;
  build_lifted_face_mask_raw(shape, landmarks, io, params, &face_mask_raw, &rm.face_contour_pts);
  int shrink_px = std::max<long long>(0, pyround(io * params.getd("face_contour_shrink_ratio", 0.02)));
  cv::Mat face_mask = shrink_px > 0 ? morph(face_mask_raw, "erode", shrink_px) : face_mask_raw.clone();

  int protect_radius = params.geti("protect_expand_radius", 6);
  double eye_extra = params.getd("eye_protect_extra_radius", 0.5);
  int brow_shrink = params.geti("brow_protect_shrink_radius", 2);
  double brow_height_ratio = params.getd("brow_protect_height_ratio", 1.0);
  double brow_width_ratio = params.getd("brow_protect_width_ratio", 1.0);
  double brow_y_shift = params.getd("brow_vertical_shift_ratio", 0.12);
  double head_yaw = estimate_head_yaw_ratio(landmarks);
  int lip_shrink = params.geti("lip_protect_shrink_radius", 2);
  int eye_radius = std::max<long long>(1, protect_radius + pyround(eye_extra));
  int lip_radius = std::max(1, protect_radius - lip_shrink);
  cv::Mat left_eye = protect_feature(shape, landmarks, LEFT_EYE, eye_radius);
  cv::Mat right_eye = protect_feature(shape, landmarks, RIGHT_EYE, eye_radius);
  cv::Mat left_brow_protect = build_brow_protect_mask(
      shape, landmarks, LEFT_BROW, LEFT_EYE, io, protect_radius, brow_shrink, brow_height_ratio,
      brow_width_ratio, brow_y_shift);
  cv::Mat right_brow_protect = build_brow_protect_mask(
      shape, landmarks, RIGHT_BROW, RIGHT_EYE, io, protect_radius, brow_shrink, brow_height_ratio,
      brow_width_ratio, brow_y_shift);
  std::vector<int> all_lips = OUTER_LIPS;
  all_lips.insert(all_lips.end(), INNER_LIPS.begin(), INNER_LIPS.end());
  cv::Mat lips = protect_feature(shape, landmarks, all_lips, lip_radius);
  cv::Mat brow_union;
  cv::bitwise_or(left_brow_protect, right_brow_protect, brow_union);
  bool protect_brows = params.getb("protect_brows", false);
  cv::Mat protect;
  cv::bitwise_or(left_eye, right_eye, protect);
  cv::bitwise_or(protect, lips, protect);
  if (protect_brows) {
    cv::bitwise_or(protect, brow_union, protect);
  } else {
    cv::Mat not_brow;
    cv::bitwise_not(brow_union, not_brow);
    cv::bitwise_and(protect, not_brow, protect);
  }
  if (params.getb("philtrum_unprotect", true)) {
    double philtrum_w = params.getd("philtrum_width_ratio", 0.16);
    protect = carve_philtrum_from_protect(protect, landmarks, io, philtrum_w);
  }
  cv::bitwise_and(protect, face_mask_raw, protect);

  Params skin_params;
  skin_params.set("skin_mask_smooth_radius", params.getd("skin_mask_smooth_radius", 3));
  cv::Mat skin = estimate_skin_mask(image_bgr, face_mask, protect, skin_params);

  // 鼻梁。
  double nose_bridge_expand = params.getd("nose_bridge_expand_ratio", 0.12);
  int bridge_thickness = std::max<long long>(2, pyround(io * nose_bridge_expand));
  cv::Mat nose_bridge = line_mask(shape, lm_pts(landmarks, NOSE_BRIDGE), bridge_thickness);
  nose_bridge = morph(nose_bridge, "dilate", std::max<long long>(1, pyround(io * 0.015)));

  // 鼻尖。
  auto nose_pts = lm_pts(landmarks, NOSE_WING);
  double nose_width, nose_height;
  if (nose_pts.size() >= 2) {
    float x0 = nose_pts[0].x, x1 = nose_pts[0].x, y0 = nose_pts[0].y, y1 = nose_pts[0].y;
    for (const auto& p : nose_pts) {
      x0 = std::min(x0, p.x);
      x1 = std::max(x1, p.x);
      y0 = std::min(y0, p.y);
      y1 = std::max(y1, p.y);
    }
    nose_width = static_cast<double>(x1 - x0);
    nose_height = static_cast<double>(y1 - y0);
  } else {
    nose_width = io * 0.45;
    nose_height = io * 0.35;
  }
  double tip_expand = params.getd("nose_tip_expand_ratio", 0.72);
  cv::Point2f tip_center = landmarks.rows > 4
                               ? cv::Point2f(landmarks.at<float>(4, 0), landmarks.at<float>(4, 1))
                               : geo.nose_tip;
  cv::Mat nose_tip = ellipse_mask(
      shape, tip_center,
      cv::Size2d(std::max(2.0, nose_width * tip_expand * 0.55), std::max(2.0, nose_height * tip_expand * 0.75)),
      geo.angle_deg);

  // 额头 + 人中。
  cv::Mat forehead = build_forehead_mask(shape, landmarks, io, face_mask_raw, params);
  forehead = clip_forehead(forehead, protect);
  cv::Mat philtrum = build_philtrum_mask(shape, landmarks, io, params);

  // 姿态自适应局部坐标。
  cv::Point2f origin = geo.nose_tip;
  cv::Mat u_grid(h, w, CV_32F), v_grid(h, w, CV_32F);
  for (int y = 0; y < h; ++y) {
    float* ug = u_grid.ptr<float>(y);
    float* vg = v_grid.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      float dx = static_cast<float>(x) - origin.x;
      float dy = static_cast<float>(y) - origin.y;
      ug[x] = dx * geo.x_axis[0] + dy * geo.x_axis[1];
      vg[x] = dx * geo.y_axis[0] + dy * geo.y_axis[1];
    }
  }
  auto oval_p = lm_pts(landmarks, FACE_OVAL);
  std::vector<float> oval_u, oval_v;
  project_points(oval_p, origin, geo.x_axis, geo.y_axis, &oval_u, &oval_v);
  double face_w_local = io * 2.0, face_h_local = io * 2.8;
  if (!oval_u.empty()) {
    float mn = oval_u[0], mx = oval_u[0];
    for (float v : oval_u) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    face_w_local = std::max(1.0, static_cast<double>(mx - mn));
  }
  if (!oval_v.empty()) {
    float mn = oval_v[0], mx = oval_v[0];
    for (float v : oval_v) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    face_h_local = std::max(1.0, static_cast<double>(mx - mn));
  }

  auto lower_eye_p = lm_pts(landmarks, LEFT_LOWER_EYE);
  {
    auto rl = lm_pts(landmarks, RIGHT_LOWER_EYE);
    lower_eye_p.insert(lower_eye_p.end(), rl.begin(), rl.end());
  }
  std::vector<float> tmp_u, lower_eye_v;
  project_points(lower_eye_p, origin, geo.x_axis, geo.y_axis, &tmp_u, &lower_eye_v);
  std::vector<int> mouth_ids = OUTER_LIPS;
  mouth_ids.insert(mouth_ids.end(), MOUTH_CORNERS.begin(), MOUTH_CORNERS.end());
  auto mouth_p = lm_pts(landmarks, mouth_ids);
  std::vector<float> mouth_v;
  project_points(mouth_p, origin, geo.x_axis, geo.y_axis, &tmp_u, &mouth_v);
  auto nosewing_p = lm_pts(landmarks, NOSE_WING);
  std::vector<float> nose_u, tmp_v;
  project_points(nosewing_p, origin, geo.x_axis, geo.y_axis, &nose_u, &tmp_v);
  std::vector<int> chin_ids = CHIN_AREA;
  chin_ids.push_back(152);
  auto chin_p = lm_pts(landmarks, chin_ids);
  std::vector<float> chin_v;
  project_points(chin_p, origin, geo.x_axis, geo.y_axis, &tmp_u, &chin_v);

  std::vector<float> lower_eye_v_copy = lower_eye_v;
  double cheek_upper = lower_eye_v.empty() ? -0.15 * face_h_local
                                           : np_median(lower_eye_v_copy, 0) - 0.05 * face_h_local;
  std::vector<float> mouth_v_copy = mouth_v;
  double cheek_lower = mouth_v.empty() ? 0.45 * face_h_local
                                       : np_percentile(mouth_v_copy, 75.0) + 0.07 * face_h_local;
  double nose_half = io * 0.12;
  if (!nose_u.empty()) {
    std::vector<float> nu1 = nose_u, nu2 = nose_u;
    nose_half = (np_percentile(nu1, 85.0) - np_percentile(nu2, 15.0)) / 2.0;
  }
  nose_half = std::max(io * 0.08, nose_half);
  double cheek_expand = params.getd("cheek_expand_ratio", 1.0);
  double cheek_margin = (cheek_expand - 1.0) * 0.06 * face_h_local;
  double midline_u = 0.0;

  cv::Mat left_cheek = cv::Mat::zeros(shape, CV_8U);
  cv::Mat right_cheek = cv::Mat::zeros(shape, CV_8U);
  {
    float cu = static_cast<float>(cheek_upper - cheek_margin);
    float cl = static_cast<float>(cheek_lower + cheek_margin);
    float lt = static_cast<float>(midline_u - nose_half * 0.75);
    float rt = static_cast<float>(midline_u + nose_half * 0.75);
    for (int y = 0; y < h; ++y) {
      const float* ug = u_grid.ptr<float>(y);
      const float* vg = v_grid.ptr<float>(y);
      const uint8_t* fm = face_mask.ptr<uint8_t>(y);
      uint8_t* lc = left_cheek.ptr<uint8_t>(y);
      uint8_t* rc = right_cheek.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) {
        bool band = vg[x] >= cu && vg[x] <= cl;
        if (band && fm[x] > 0) {
          if (ug[x] < lt) lc[x] = 255;
          if (ug[x] > rt) rc[x] = 255;
        }
      }
    }
  }

  std::vector<float> mouth_v_copy2 = mouth_v;
  double lower_lip_v = mouth_v.empty() ? 0.35 * face_h_local : np_percentile(mouth_v_copy2, 90.0);
  double chin_upper = lower_lip_v - 0.01 * face_h_local;
  double chin_lower = 0.70 * face_h_local;
  if (!chin_v.empty()) {
    float mx = chin_v[0];
    for (float v : chin_v) mx = std::max(mx, v);
    chin_lower = static_cast<double>(mx) + 0.02 * face_h_local;
  }
  cv::Mat chin = cv::Mat::zeros(shape, CV_8U);
  {
    float cu = static_cast<float>(chin_upper), cl = static_cast<float>(chin_lower);
    for (int y = 0; y < h; ++y) {
      const float* vg = v_grid.ptr<float>(y);
      const uint8_t* fm = face_mask.ptr<uint8_t>(y);
      uint8_t* c = chin.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x)
        if (vg[x] >= cu && vg[x] <= cl && fm[x] > 0) c[x] = 255;
    }
  }
  double chin_expand = params.getd("chin_expand_ratio", 1.0);
  if (chin_expand > 1.01) chin = morph(chin, "dilate", pyround(io * 0.04 * (chin_expand - 1.0)));

  nose_bridge = clip_to_skin(nose_bridge, skin, protect);
  nose_tip = clip_to_skin(nose_tip, skin, protect);
  chin = clip_to_skin(chin, skin, protect);
  left_cheek = clip_to_skin(left_cheek, skin, protect);
  right_cheek = clip_to_skin(right_cheek, skin, protect);
  philtrum = clip_forehead(philtrum, protect);

  // 眉弓（剩余集合）。
  cv::Mat priority;
  cv::bitwise_or(left_cheek, right_cheek, priority);
  cv::bitwise_or(forehead, priority, priority);
  cv::Mat tmp;
  cv::bitwise_or(left_eye, right_eye, tmp);
  cv::bitwise_or(priority, tmp, priority);
  cv::bitwise_or(left_brow_protect, right_brow_protect, tmp);
  cv::bitwise_or(priority, tmp, priority);
  cv::bitwise_or(nose_bridge, nose_tip, tmp);
  cv::bitwise_or(priority, tmp, priority);

  std::function<bool(int, int)> left_side, right_side;
  if (std::abs(head_yaw) > 0.12) {
    float eye_mid_x = geo.eye_mid.x;
    float lo = static_cast<float>(eye_mid_x - io * 0.04);
    float hi = static_cast<float>(eye_mid_x + io * 0.04);
    left_side = [lo](int x, int) { return static_cast<float>(x) < lo; };
    right_side = [hi](int x, int) { return static_cast<float>(x) > hi; };
  } else {
    float lt = static_cast<float>(midline_u - nose_half * 0.35);
    float rt = static_cast<float>(midline_u + nose_half * 0.35);
    const cv::Mat ug = u_grid;
    left_side = [ug, lt](int x, int y) { return ug.at<float>(y, x) < lt; };
    right_side = [ug, rt](int x, int y) { return ug.at<float>(y, x) > rt; };
  }
  cv::Mat left_brow_ridge = build_residual_brow_ridge_side(
      shape, lm_pts(landmarks, LEFT_BROW), lm_pts(landmarks, LEFT_EYE), io, face_mask, skin,
      priority, left_side, brow_height_ratio, brow_width_ratio, protect_radius, brow_shrink,
      brow_y_shift);
  cv::Mat right_brow_ridge = build_residual_brow_ridge_side(
      shape, lm_pts(landmarks, RIGHT_BROW), lm_pts(landmarks, RIGHT_EYE), io, face_mask, skin,
      priority, right_side, brow_height_ratio, brow_width_ratio, protect_radius, brow_shrink,
      brow_y_shift);
  cv::Mat brow_ridge;
  cv::bitwise_or(left_brow_ridge, right_brow_ridge, brow_ridge);
  cv::Mat not_ridge;
  cv::bitwise_not(brow_ridge, not_ridge);
  cv::bitwise_and(protect, not_ridge, protect);
  cv::bitwise_and(forehead, not_ridge, forehead);

  // 皮肤 mask 必须覆盖整个额头候选区。
  cv::bitwise_or(skin, forehead, skin);

  // 区域互斥清理。
  cv::Mat non_cheek;
  cv::bitwise_or(nose_bridge, nose_tip, non_cheek);
  cv::bitwise_or(forehead, chin, tmp);
  cv::bitwise_or(non_cheek, tmp, non_cheek);
  cv::bitwise_or(non_cheek, philtrum, non_cheek);
  cv::bitwise_or(non_cheek, brow_ridge, non_cheek);
  cv::bitwise_or(non_cheek, protect, non_cheek);
  left_cheek.setTo(0, non_cheek);
  right_cheek.setTo(0, non_cheek);

  cv::Mat candidate = cv::Mat::zeros(shape, CV_8U);
  for (const cv::Mat& m : {nose_bridge, nose_tip, forehead, left_cheek, right_cheek, chin, philtrum,
                           left_brow_ridge, right_brow_ridge})
    cv::bitwise_or(candidate, m, candidate);

  if (cv::countNonZero(skin) == 0)
    rm.warnings.push_back("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点");
  if (cv::countNonZero(candidate) == 0)
    rm.warnings.push_back("高光候选区域为空，请检查区域扩张参数");
  if (std::abs(head_yaw) > 0.18) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "侧脸角度较大（yaw≈%.2f），分区可能不准；建议使用正脸照片，或开启「人脸轻量对齐」", head_yaw);
    rm.warnings.push_back(buf);
  } else if (std::abs(head_yaw) > 0.12) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "检测到轻微侧脸（yaw≈%.2f），眉/额分区已按左右分别拟合", head_yaw);
    rm.warnings.push_back(buf);
  }

  rm.masks["face_mask_raw"] = face_mask_raw;
  rm.masks["face_mask"] = face_mask;
  rm.masks["skin"] = skin;
  rm.masks["protect"] = protect;
  rm.masks["left_eye"] = left_eye;
  rm.masks["right_eye"] = right_eye;
  rm.masks["left_brow_protect"] = left_brow_protect;
  rm.masks["right_brow_protect"] = right_brow_protect;
  rm.masks["left_brow_ridge"] = left_brow_ridge;
  rm.masks["right_brow_ridge"] = right_brow_ridge;
  rm.masks["lips"] = lips;
  rm.masks["mouth"] = lips;
  rm.masks["nose_bridge"] = nose_bridge;
  rm.masks["nose_tip"] = nose_tip;
  rm.masks["forehead"] = forehead;
  rm.masks["left_cheek"] = left_cheek;
  rm.masks["right_cheek"] = right_cheek;
  rm.masks["chin"] = chin;
  rm.masks["philtrum"] = philtrum;
  rm.masks["highlight_candidate"] = candidate;
  return rm;
}

}  // namespace facehi
