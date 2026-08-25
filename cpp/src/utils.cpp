#include "high_removal/common.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace hr {

long long pyRound(double x) {
    // Python round()：half-to-even。
    const double floorV = std::floor(x);
    const double diff = x - floorV;
    if (diff > 0.5) return static_cast<long long>(floorV) + 1;
    if (diff < 0.5) return static_cast<long long>(floorV);
    const long long f = static_cast<long long>(floorV);
    return (f % 2 == 0) ? f : f + 1;
}

double percentileOf(std::vector<float> values, double q, double fallback) {
    if (values.empty()) return fallback;
    std::sort(values.begin(), values.end());
    const double pos = (static_cast<double>(values.size()) - 1.0) * q / 100.0;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return static_cast<double>(values[lo]);
    const double frac = pos - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - frac) + static_cast<double>(values[hi]) * frac;
}

double medianOf(std::vector<float> values, double fallback) {
    return percentileOf(std::move(values), 50.0, fallback);
}

float kthLargest(std::vector<float>& vals, int k) {
    // np.partition(vals, -k)[-k]
    const size_t pos = vals.size() - static_cast<size_t>(k);
    std::nth_element(vals.begin(), vals.begin() + pos, vals.end());
    return vals[pos];
}

std::vector<float> interp1d(const std::vector<float>& x, const std::vector<double>& xp, const std::vector<double>& fp) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const double xi = x[i];
        if (xi <= xp.front()) {
            out[i] = static_cast<float>(fp.front());
        } else if (xi >= xp.back()) {
            out[i] = static_cast<float>(fp.back());
        } else {
            const auto it = std::upper_bound(xp.begin(), xp.end(), xi);
            const size_t hi = static_cast<size_t>(it - xp.begin());
            const size_t lo = hi - 1;
            const double t = (xi - xp[lo]) / (xp[hi] - xp[lo]);
            out[i] = static_cast<float>(fp[lo] + t * (fp[hi] - fp[lo]));
        }
    }
    return out;
}

std::vector<float> polyfit2Eval(const std::vector<double>& bx, const std::vector<double>& by, int width) {
    // np.polyfit(bx, by, 2)：最小二乘（SVD），再 polyval。
    const int n = static_cast<int>(bx.size());
    cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        A.at<double>(i, 0) = bx[i] * bx[i];
        A.at<double>(i, 1) = bx[i];
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = by[i];
    }
    cv::Mat coef;
    cv::solve(A, b, coef, cv::DECOMP_SVD);
    const double c2 = coef.at<double>(0), c1 = coef.at<double>(1), c0 = coef.at<double>(2);
    std::vector<float> out(width);
    for (int x = 0; x < width; ++x) {
        const double xv = static_cast<double>(x);
        out[x] = static_cast<float>(c2 * xv * xv + c1 * xv + c0);
    }
    return out;
}

cv::Mat morphMask(const cv::Mat& mask, const std::string& op, double radius) {
    const int r = static_cast<int>(std::max<long long>(0, pyRound(radius)));
    if (r <= 0) return mask.clone();
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r + 1, 2 * r + 1));
    cv::Mat out;
    if (op == "dilate") {
        cv::dilate(mask, out, k);
    } else if (op == "erode") {
        cv::erode(mask, out, k);
    } else if (op == "open") {
        cv::morphologyEx(mask, out, cv::MORPH_OPEN, k);
    } else if (op == "close") {
        cv::morphologyEx(mask, out, cv::MORPH_CLOSE, k);
    } else {
        CV_Error(cv::Error::StsBadArg, "未知形态学操作");
    }
    return out;
}

cv::Mat blurMask(const cv::Mat& mask, double radius) {
    const int r = static_cast<int>(std::max<long long>(0, pyRound(radius)));
    if (r <= 0) return mask.clone();
    cv::Mat out;
    cv::GaussianBlur(mask, out, cv::Size(2 * r + 1, 2 * r + 1), 0);
    return out;
}

static std::vector<cv::Point> roundPts(const std::vector<cv::Point2f>& pts) {
    // np.round(...).astype(np.int32)：银行家舍入。
    std::vector<cv::Point> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.emplace_back(static_cast<int>(pyRound(p.x)), static_cast<int>(pyRound(p.y)));
    }
    return out;
}

cv::Mat fillPolyMask(cv::Size size, const std::vector<cv::Point2f>& pts) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    if (pts.size() < 3) return mask;
    std::vector<std::vector<cv::Point>> polys{roundPts(pts)};
    cv::fillPoly(mask, polys, cv::Scalar(255));
    return mask;
}

cv::Mat fillHullMask(cv::Size size, const std::vector<cv::Point2f>& pts) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    if (pts.size() < 3) return mask;
    std::vector<cv::Point> hull;
    cv::convexHull(roundPts(pts), hull);
    cv::fillConvexPoly(mask, hull, cv::Scalar(255));
    return mask;
}

cv::Mat lineMask(cv::Size size, const std::vector<cv::Point2f>& pts, int thickness) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    if (pts.size() < 2) return mask;
    std::vector<std::vector<cv::Point>> polys{roundPts(pts)};
    cv::polylines(mask, polys, false, cv::Scalar(255), std::max(1, thickness), cv::LINE_AA);
    return mask;
}

cv::Mat ellipseMask(cv::Size size, cv::Point2f center, cv::Size2f axes, double angleDeg) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    const cv::Point c(static_cast<int>(pyRound(center.x)), static_cast<int>(pyRound(center.y)));
    const cv::Size ax(std::max(1, static_cast<int>(pyRound(axes.width))), std::max(1, static_cast<int>(pyRound(axes.height))));
    cv::ellipse(mask, c, ax, angleDeg, 0, 360, cv::Scalar(255), -1, cv::LINE_AA);
    return mask;
}

std::optional<Bbox> computeRoiBbox(const cv::Mat& hard, const cv::Mat& soft, cv::Size shape, double paddingRatio) {
    int x0 = INT32_MAX, y0 = INT32_MAX, x1 = -1, y1 = -1;
    for (int y = 0; y < hard.rows; ++y) {
        const uchar* ph = hard.ptr<uchar>(y);
        const uchar* ps = soft.ptr<uchar>(y);
        for (int x = 0; x < hard.cols; ++x) {
            if (ph[x] > 0 || ps[x] > 0) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
        }
    }
    if (x1 < 0) return std::nullopt;
    const int w = shape.width, h = shape.height;
    const int padX = std::max(8, static_cast<int>((x1 - x0 + 1) * paddingRatio));
    const int padY = std::max(8, static_cast<int>((y1 - y0 + 1) * paddingRatio));
    x0 = std::max(0, x0 - padX);
    y0 = std::max(0, y0 - padY);
    x1 = std::min(w - 1, x1 + padX);
    y1 = std::min(h - 1, y1 + padY);
    return Bbox{x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

cv::Mat resizeForProcess(const cv::Mat& img, double scale) {
    scale = clampProcessScale(scale);
    if (scale >= 0.999) return img;
    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

cv::Mat upsampleMaskHard(const cv::Mat& mask, cv::Size size) {
    cv::Mat up;
    cv::resize(mask, up, size, 0, 0, cv::INTER_NEAREST);
    return up;
}

cv::Mat upsampleMaskSoft(const cv::Mat& mask, cv::Size size) {
    cv::Mat up;
    cv::resize(mask, up, size, 0, 0, cv::INTER_LINEAR);
    cv::GaussianBlur(up, up, cv::Size(3, 3), 0);
    return up;
}

Section scaledHighlightParams(const Section& params, double scale) {
    scale = clampProcessScale(scale);
    if (scale >= 0.999) return params;
    Section out = params;
    const double baseMin = getF(params, {"highlight_min_area"}, 8);
    if (scale <= 0.26) {
        out["highlight_min_area"] = static_cast<double>(std::max<long long>(2, pyRound(baseMin * scale)));
    } else {
        out["highlight_min_area"] = static_cast<double>(std::max<long long>(1, pyRound(baseMin * scale * scale)));
    }
    for (const char* key : {"mask_dilate_radius", "mask_erode_radius", "mask_blur_radius", "morph_close_radius"}) {
        auto it = params.find(key);
        if (it == params.end()) continue;
        const double base = getF(params, {key}, 0);
        const long long scaled = pyRound(base * scale);
        long long floorV = 0;
        const std::string k(key);
        if (k == "morph_close_radius" || k == "mask_dilate_radius") floorV = 1;
        if (k == "mask_blur_radius") floorV = (scale <= 0.26) ? 3 : 0;
        out[key] = static_cast<double>(std::max(floorV, scaled));
    }
    return out;
}

double clampProcessScale(double scale) {
    return std::max(0.25, std::min(1.0, scale));
}

ProcessScales resolveProcessScales(const Section& pipeline) {
    // resolve_process_scales()：字符串 compromise/折中/0.25_0.5 → (0.25, 0.5)。
    auto it = pipeline.find("process_scale");
    if (it != pipeline.end() && std::holds_alternative<std::string>(it->second)) {
        std::string raw = std::get<std::string>(it->second);
        std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char c) { return std::tolower(c); });
        if (raw == "compromise" || raw == "折中" || raw == "0.25_0.5") {
            return {0.25, 0.5, true};
        }
        const double v = clampProcessScale(std::stod(raw));
        return {v, v, false};
    }
    const double v = clampProcessScale(getF(pipeline, {"process_scale"}, 1.0));
    return {v, v, false};
}

}  // namespace hr
