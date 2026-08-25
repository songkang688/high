// 公共类型与工具。与 highlight_removal/utils.py 中的辅助函数一一对应，
// 并额外提供与 NumPy 语义一致的统计函数（percentile / median / 第 k 大阈值 / banker 舍入）。
#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace hr {

using Landmarks = std::vector<cv::Vec3f>;  // 478 x (x, y, z)，像素坐标

using Value = std::variant<double, bool, std::string>;
using Section = std::map<std::string, Value>;

struct Config {
    Section face_detection;
    Section regions;
    Section highlight_detection;
    Section highlight_removal;
    Section pipeline;
};

Config loadConfig(const std::string& path);

// 参数读取：按名字列表顺序找第一个存在的键（对应 Python 的 _p 别名兼容逻辑）。
double getF(const Section& s, std::initializer_list<const char*> names, double def);
bool getB(const Section& s, std::initializer_list<const char*> names, bool def);
std::string getS(const Section& s, std::initializer_list<const char*> names, const std::string& def);

// ---- NumPy 语义 ----

// Python round()/np.round()：银行家舍入（half-to-even），与 std::round 不同。
long long pyRound(double x);

// np.percentile(values, q)，线性插值；values 为 float32 存储。
double percentileOf(std::vector<float> values, double q, double fallback);

// np.median：偶数个取中间两数平均。
double medianOf(std::vector<float> values, double fallback);

// np.partition(vals, -k)[-k]：第 k 大的值（vals 会被打乱）。
float kthLargest(std::vector<float>& vals, int k);

// np.interp(x, xp, fp)：分段线性，端点外取端值（xp 需升序）。
std::vector<float> interp1d(const std::vector<float>& x, const std::vector<double>& xp, const std::vector<double>& fp);

// np.polyfit(x, y, 2) + np.polyval：双精度最小二乘二次拟合后在 x 处取值。
std::vector<float> polyfit2Eval(const std::vector<double>& bx, const std::vector<double>& by, int width);

// ---- 掩码工具（utils.py）----

cv::Mat morphMask(const cv::Mat& mask, const std::string& op, double radius);   // morph()
cv::Mat blurMask(const cv::Mat& mask, double radius);                            // blur_mask()
cv::Mat fillPolyMask(cv::Size size, const std::vector<cv::Point2f>& pts);        // fill_poly_mask()
cv::Mat fillHullMask(cv::Size size, const std::vector<cv::Point2f>& pts);        // fill_hull_mask()
cv::Mat lineMask(cv::Size size, const std::vector<cv::Point2f>& pts, int thickness);  // line_mask()
cv::Mat ellipseMask(cv::Size size, cv::Point2f center, cv::Size2f axes, double angleDeg);  // ellipse_mask()

struct Bbox {
    int x = 0, y = 0, w = 0, h = 0;
};

// compute_roi_bbox()：hard|soft 的外接框 + padding。
std::optional<Bbox> computeRoiBbox(const cv::Mat& hard, const cv::Mat& soft, cv::Size shape, double paddingRatio);

cv::Mat resizeForProcess(const cv::Mat& img, double scale);   // resize_for_process()
cv::Mat upsampleMaskHard(const cv::Mat& mask, cv::Size size); // upsample_mask_hard()
cv::Mat upsampleMaskSoft(const cv::Mat& mask, cv::Size size); // upsample_mask_soft()

// scaled_highlight_params()：低分辨率检测时按比例缩放阈值参数。
Section scaledHighlightParams(const Section& params, double scale);

// clamp_process_scale / resolve_process_scales
double clampProcessScale(double scale);
struct ProcessScales {
    double face = 1.0;
    double highlight = 1.0;
    bool compromise = false;
};
ProcessScales resolveProcessScales(const Section& pipeline);

// float→uint8 截断（NumPy astype(np.uint8) 的语义，非四舍五入），输入需先 clip 到 [0,255]。
inline uchar truncU8(float v) {
    if (v <= 0.f) return 0;
    if (v >= 255.f) return 255;
    return static_cast<uchar>(v);  // 正数截断 = floor
}

}  // namespace hr
