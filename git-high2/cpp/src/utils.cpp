// 对应 highlight_removal/utils.py。
#include "facehi/utils.hpp"

#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace facehi {

cv::Mat normalize_mask(const cv::Mat& mask_u8) {
  CV_Assert(mask_u8.type() == CV_8U);
  cv::Mat out(mask_u8.size(), CV_32F);
  for (int y = 0; y < mask_u8.rows; ++y) {
    const uint8_t* m = mask_u8.ptr<uint8_t>(y);
    float* o = out.ptr<float>(y);
    for (int x = 0; x < mask_u8.cols; ++x) o[x] = static_cast<float>(m[x]) / 255.0f;
  }
  return out;
}

cv::Mat morph(const cv::Mat& mask_u8, const std::string& op, double radius) {
  int r = std::max<long long>(0, pyround(radius));
  cv::Mat m = mask_u8;
  if (r <= 0) return m.clone();
  cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1));
  cv::Mat out;
  if (op == "dilate")
    cv::dilate(m, out, k);
  else if (op == "erode")
    cv::erode(m, out, k);
  else if (op == "open")
    cv::morphologyEx(m, out, cv::MORPH_OPEN, k);
  else if (op == "close")
    cv::morphologyEx(m, out, cv::MORPH_CLOSE, k);
  else
    throw std::runtime_error("未知形态学操作: " + op);
  return out;
}

cv::Mat blur_mask(const cv::Mat& mask_u8, double radius) {
  int r = std::max<long long>(0, pyround(radius));
  if (r <= 0) return mask_u8.clone();
  cv::Mat out;
  int k = 2 * r + 1;
  cv::GaussianBlur(mask_u8, out, cv::Size(k, k), 0);
  return out;
}

static std::vector<cv::Point> round_points(const std::vector<cv::Point2f>& points) {
  std::vector<cv::Point> out;
  out.reserve(points.size());
  for (const auto& p : points) {
    // np.round：half-to-even。
    out.emplace_back(static_cast<int>(pyround(p.x)), static_cast<int>(pyround(p.y)));
  }
  return out;
}

cv::Mat fill_poly_mask(const cv::Size& size, const std::vector<cv::Point2f>& points) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8U);
  if (points.size() < 3) return mask;
  std::vector<std::vector<cv::Point>> polys{round_points(points)};
  cv::fillPoly(mask, polys, 255);
  return mask;
}

cv::Mat fill_hull_mask(const cv::Size& size, const std::vector<cv::Point2f>& points) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8U);
  if (points.size() < 3) return mask;
  std::vector<cv::Point> pts = round_points(points);
  std::vector<cv::Point> hull;
  cv::convexHull(pts, hull);
  cv::fillConvexPoly(mask, hull, 255);
  return mask;
}

cv::Mat line_mask(const cv::Size& size, const std::vector<cv::Point2f>& points, int thickness) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8U);
  if (points.size() < 2) return mask;
  std::vector<std::vector<cv::Point>> polys{round_points(points)};
  cv::polylines(mask, polys, false, 255, std::max(1, thickness), cv::LINE_AA);
  return mask;
}

cv::Mat ellipse_mask(const cv::Size& size, const cv::Point2f& center, const cv::Size2d& axes, double angle) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8U);
  int cx = static_cast<int>(pyround(center.x));
  int cy = static_cast<int>(pyround(center.y));
  int ax1 = std::max<long long>(1, pyround(axes.width));
  int ax2 = std::max<long long>(1, pyround(axes.height));
  cv::ellipse(mask, cv::Point(cx, cy), cv::Size(ax1, ax2), angle, 0, 360, 255, -1, cv::LINE_AA);
  return mask;
}

double clamp_process_scale(double scale) { return std::max(0.25, std::min(1.0, scale)); }

std::tuple<double, double, bool> resolve_process_scales(const Params& pipeline) {
  std::string raw = pipeline.gets("process_scale", "1.0");
  std::string lowered = raw;
  for (auto& c : lowered) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lowered == "compromise" || raw == "折中" || raw == "0.25_0.5")
    return {0.25, 0.5, true};
  double v = 1.0;
  try {
    v = std::stod(raw);
  } catch (...) {
    v = pipeline.getd("process_scale", 1.0);
  }
  double s = clamp_process_scale(v);
  return {s, s, false};
}

cv::Mat resize_for_process(const cv::Mat& image, double scale) {
  scale = clamp_process_scale(scale);
  if (scale >= 0.999) return image;
  cv::Mat out;
  cv::resize(image, out, cv::Size(), scale, scale, cv::INTER_AREA);
  return out;
}

cv::Mat scale_landmarks(const cv::Mat& landmarks, double scale) {
  cv::Mat lm = landmarks.clone();
  float s = static_cast<float>(scale);
  for (int i = 0; i < lm.rows; ++i) {
    lm.at<float>(i, 0) *= s;
    lm.at<float>(i, 1) *= s;
    if (lm.cols > 2) lm.at<float>(i, 2) *= s;
  }
  return lm;
}

cv::Mat upsample_mask_hard(const cv::Mat& mask_u8, const cv::Size& size) {
  cv::Mat up;
  cv::resize(mask_u8, up, size, 0, 0, cv::INTER_NEAREST);
  return up;
}

cv::Mat upsample_mask_soft(const cv::Mat& mask_u8, const cv::Size& size) {
  cv::Mat up;
  cv::resize(mask_u8, up, size, 0, 0, cv::INTER_LINEAR);
  cv::Mat out;
  cv::GaussianBlur(up, out, cv::Size(3, 3), 0);
  return out;
}

Params scaled_highlight_params(const Params& params, double scale) {
  scale = clamp_process_scale(scale);
  if (scale >= 0.999) return params;
  Params out = params;
  double base_min = params.getd("highlight_min_area", 8);
  if (scale <= 0.26)
    out.set("highlight_min_area", static_cast<double>(std::max<long long>(2, pyround(base_min * scale))));
  else
    out.set("highlight_min_area", static_cast<double>(std::max<long long>(1, pyround(base_min * scale * scale))));
  for (const std::string& key : {"mask_dilate_radius", "mask_erode_radius", "mask_blur_radius", "morph_close_radius"}) {
    if (!params.has(key)) continue;
    double base = params.getd(key.c_str(), 0.0);
    long long scaled = pyround(base * scale);
    long long floor_v = (key == "morph_close_radius" || key == "mask_dilate_radius") ? 1 : 0;
    if (key == "mask_blur_radius") floor_v = (scale <= 0.26) ? 3 : 0;
    out.set(key, static_cast<double>(std::max(floor_v, scaled)));
  }
  return out;
}

std::optional<cv::Rect> compute_roi_bbox(const cv::Mat& hard_u8, const cv::Mat& soft_u8,
                                         const cv::Size& size, double padding_ratio) {
  int min_x = INT_MAX, min_y = INT_MAX, max_x = -1, max_y = -1;
  for (int y = 0; y < hard_u8.rows; ++y) {
    const uint8_t* h = hard_u8.ptr<uint8_t>(y);
    const uint8_t* s = soft_u8.ptr<uint8_t>(y);
    for (int x = 0; x < hard_u8.cols; ++x) {
      if (h[x] > 0 || s[x] > 0) {
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
      }
    }
  }
  if (max_x < 0) return std::nullopt;
  int w = size.width, h = size.height;
  int pad_x = std::max(8, static_cast<int>((max_x - min_x + 1) * padding_ratio));
  int pad_y = std::max(8, static_cast<int>((max_y - min_y + 1) * padding_ratio));
  int x0 = std::max(0, min_x - pad_x);
  int y0 = std::max(0, min_y - pad_y);
  int x1 = std::min(w - 1, max_x + pad_x);
  int y1 = std::min(h - 1, max_y + pad_y);
  return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

cv::Mat read_image_bgr(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.empty()) return {};
  cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
  if (img.empty()) img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
  if (img.empty()) return {};
  if (img.channels() == 1) {
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }
  if (img.channels() == 4) {
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
  }
  return img;
}

bool write_image(const std::string& path, const cv::Mat& image_bgr) {
  std::string ext = ".png";
  auto pos = path.find_last_of('.');
  if (pos != std::string::npos) {
    ext = path.substr(pos);
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const std::vector<std::string> allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"};
    bool ok = false;
    for (const auto& a : allowed)
      if (ext == a) ok = true;
    if (!ok) ext = ".png";
  }
  std::vector<uint8_t> buf;
  if (!cv::imencode(ext, image_bgr, buf)) return false;
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  return static_cast<bool>(f);
}

}  // namespace facehi
