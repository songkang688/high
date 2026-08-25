#include "high_removal/regions.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

#include "high_removal/topology.hpp"

namespace hr {

namespace {

using topo::CHIN_AREA;
using topo::FACE_OVAL;
using topo::INNER_LIPS;
using topo::LEFT_BROW;
using topo::LEFT_EYE;
using topo::LEFT_LOWER_EYE;
using topo::MOUTH_CORNERS;
using topo::NOSE_BRIDGE;
using topo::NOSE_TIP;
using topo::NOSE_WING;
using topo::OUTER_LIPS;
using topo::RIGHT_BROW;
using topo::RIGHT_EYE;
using topo::RIGHT_LOWER_EYE;

int countNonZero8(const cv::Mat& m) { return cv::countNonZero(m); }

cv::Mat orMask(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat out;
    cv::bitwise_or(a, b, out);
    return out;
}

// _protect_feature()
cv::Mat protectFeature(cv::Size size, const Landmarks& lm, const std::vector<int>& indices, int dilatePx) {
    cv::Mat mask = fillHullMask(size, pickPts(lm, indices));
    if (dilatePx > 0) mask = morphMask(mask, "dilate", dilatePx);
    return mask;
}

std::vector<cv::Point2f> vstack(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    std::vector<cv::Point2f> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

std::vector<float> ys(const std::vector<cv::Point2f>& p) {
    std::vector<float> out;
    out.reserve(p.size());
    for (const auto& q : p) out.push_back(q.y);
    return out;
}

// _smooth_brow_arc_curve()
struct ArcCurve {
    std::vector<float> y;  // 长度 = 宽度
    double xLeft = 0.0;
    double xRight = 0.0;
};

ArcCurve smoothBrowArcCurve(cv::Size size, const std::vector<cv::Point2f>& browPts, double io,
                            double widthRatio, double yShiftRatio) {
    const int h = size.height, w = size.width;
    ArcCurve arc;
    if (browPts.size() < 2) {
        arc.y.assign(w, static_cast<float>(h));
        arc.xLeft = 0.0;
        arc.xRight = w - 1.0;
        return arc;
    }
    std::vector<cv::Point2f> sorted = browPts;
    std::stable_sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    std::vector<double> bx, by;
    for (const auto& p : sorted) {
        bx.push_back(p.x);
        by.push_back(p.y);
    }
    std::vector<float> y;
    if (bx.size() >= 3) {
        y = polyfit2Eval(bx, by, w);
    } else {
        std::vector<float> xx(w);
        for (int i = 0; i < w; ++i) xx[i] = static_cast<float>(i);
        y = interp1d(xx, bx, by);
    }
    long long k = std::max<long long>(11, pyRound(io * 0.34));
    if (k % 2 == 0) ++k;
    cv::Mat row(1, w, CV_32F, y.data());
    cv::Mat blurred;
    cv::GaussianBlur(row, blurred, cv::Size(static_cast<int>(k), 1), 0);
    arc.y.resize(w);
    const float shift = static_cast<float>(io * std::max(0.0, yShiftRatio));
    for (int i = 0; i < w; ++i) arc.y[i] = blurred.at<float>(0, i) - shift;
    const double ext = io * 0.028 * std::max(0.5, widthRatio);
    arc.xLeft = bx.front() - ext;
    arc.xRight = bx.back() + ext;
    return arc;
}

// _arc_band_mask()
cv::Mat arcBandMask(cv::Size size, const std::vector<float>& yTop, const std::vector<float>& yBot,
                    double xLeft, double xRight) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < size.height; ++y) {
        uchar* row = mask.ptr<uchar>(y);
        const float yf = static_cast<float>(y);
        for (int x = 0; x < size.width; ++x) {
            if (x >= xLeft && x <= xRight && yf >= yTop[x] && yf <= yBot[x]) row[x] = 255;
        }
    }
    return mask;
}

// _brow_protect_thickness()
std::pair<double, double> browProtectThickness(double io, int dilatePx, int shrinkPx, double heightRatio) {
    const double hRatio = std::max(0.5, heightRatio);
    const double padUp = io * 0.016 * hRatio;
    double padDown = io * 0.009 * hRatio;
    padDown += std::max(0, dilatePx - shrinkPx) * io * 0.004 * hRatio;
    return {padUp, padDown};
}

// _build_brow_protect_mask()
cv::Mat buildBrowProtectMask(cv::Size size, const Landmarks& lm, const std::vector<int>& browIdx,
                             const std::vector<int>& eyeIdx, double io, int dilatePx, int shrinkPx,
                             double heightRatio, double widthRatio, double yShiftRatio) {
    const auto eyePts = pickPts(lm, eyeIdx);
    const auto brow = filterSideBrowPts(pickPts(lm, browIdx), eyePts, io);
    if (brow.size() < 2 || eyePts.size() < 2) return cv::Mat::zeros(size, CV_8U);
    float ex0 = eyePts[0].x, ex1 = eyePts[0].x;
    for (const auto& p : eyePts) {
        ex0 = std::min(ex0, p.x);
        ex1 = std::max(ex1, p.x);
    }
    if (ex1 - ex0 < io * 0.07) return cv::Mat::zeros(size, CV_8U);
    const ArcCurve arc = smoothBrowArcCurve(size, brow, io, widthRatio, yShiftRatio);
    const auto [padUp, padDown] = browProtectThickness(io, dilatePx, shrinkPx, heightRatio);
    std::vector<float> yTop(arc.y.size()), yBot(arc.y.size());
    for (size_t i = 0; i < arc.y.size(); ++i) {
        yTop[i] = arc.y[i] - static_cast<float>(padUp);
        yBot[i] = arc.y[i] + static_cast<float>(padDown);
    }
    return arcBandMask(size, yTop, yBot, arc.xLeft, arc.xRight);
}

// _compute_forehead_lift_px()
double computeForeheadLiftPx(const Landmarks& lm, double io, const std::vector<cv::Point2f>& ovalPts, double fhExpand) {
    const auto browPts = vstack(pickPts(lm, LEFT_BROW), pickPts(lm, RIGHT_BROW));
    if (browPts.size() < 2 || ovalPts.size() < 3) return io * 0.20;
    const double browBottomY = percentileOf(ys(browPts), 72, 0);
    float ovalMinY = ovalPts[0].y;
    for (const auto& p : ovalPts) ovalMinY = std::min(ovalMinY, p.y);
    const double span = std::max({browBottomY - ovalMinY, io * 0.20, 4.0});
    if (fhExpand <= 1.0) return io * 0.20 * std::max(0.35, fhExpand);
    return io * 0.20 + (fhExpand - 1.0) * span;
}

// _lift_face_oval_pts()
std::vector<cv::Point2f> liftFaceOvalPts(const Landmarks& lm, double io, double foreheadHeightRatio) {
    auto ovalPts = pickPts(lm, FACE_OVAL);
    if (ovalPts.size() < 3) return ovalPts;
    const auto browPts = vstack(pickPts(lm, LEFT_BROW), pickPts(lm, RIGHT_BROW));
    double browBottomY;
    if (browPts.size() >= 2) {
        browBottomY = percentileOf(ys(browPts), 72, 0);
    } else {
        float minY = ovalPts[0].y;
        for (const auto& p : ovalPts) minY = std::min(minY, p.y);
        browBottomY = minY + io * 0.35;
    }
    const double liftPx = computeForeheadLiftPx(lm, io, ovalPts, foreheadHeightRatio);
    if (liftPx <= 0.5) return ovalPts;
    for (auto& p : ovalPts) {
        if (p.y < browBottomY - 1.0) p.y = std::max(0.0f, p.y - static_cast<float>(liftPx));
    }
    return ovalPts;
}

// _forehead_support_oval()
cv::Mat foreheadSupportOval(cv::Size size, const Landmarks& lm, const cv::Mat& faceMaskRaw,
                            double liftPx, double browBottomY) {
    if (liftPx <= 0.5) return faceMaskRaw;
    auto ovalPts = pickPts(lm, FACE_OVAL);
    if (ovalPts.size() < 3) return faceMaskRaw;
    for (auto& p : ovalPts) {
        if (p.y < browBottomY - 1.0) p.y = std::max(0.0f, p.y - static_cast<float>(liftPx));
    }
    cv::Mat out = fillPolyMask(size, ovalPts);
    if (countNonZero8(out) == 0) out = fillHullMask(size, ovalPts);
    return countNonZero8(out) ? out : faceMaskRaw;
}

// _build_lifted_face_mask_raw()
cv::Mat buildLiftedFaceMaskRaw(cv::Size size, const Landmarks& lm, double io, const Section& params) {
    const double fhExpand = getF(params, {"forehead_height_ratio"}, 1.28);
    auto contourPts = liftFaceOvalPts(lm, io, fhExpand);
    cv::Mat mask = fillPolyMask(size, contourPts);
    if (countNonZero8(mask) == 0) mask = fillHullMask(size, contourPts);
    if (countNonZero8(mask) == 0) {
        contourPts = pickPts(lm, FACE_OVAL);
        mask = fillPolyMask(size, contourPts);
        if (countNonZero8(mask) == 0) mask = fillHullMask(size, contourPts);
    }
    return mask;
}

// _philtrum_box()：返回 y0, y1, x0, x1（无效时 y1<=y0）。
struct PhiltrumBox {
    int y0 = 0, y1 = 0, x0 = 0, x1 = 0;
    bool valid = false;
};

PhiltrumBox philtrumBox(cv::Size size, const Landmarks& lm, double io, double widthRatio) {
    PhiltrumBox box;
    const auto upperLip = pickPts(lm, {61, 0, 37, 39, 40, 185, 291, 375, 321, 405, 314, 17, 267, 269, 270});
    const auto noseBottom = pickPts(lm, {2, 94, 19, 4, 5, 98, 327, 326, 49, 279});
    if (upperLip.size() < 2 || noseBottom.size() < 2) return box;
    float lipTopY = upperLip[0].y;
    double cxSum = 0;
    for (const auto& p : upperLip) {
        lipTopY = std::min(lipTopY, p.y);
        cxSum += p.x;
    }
    float noseBaseY = noseBottom[0].y;
    for (const auto& p : noseBottom) noseBaseY = std::max(noseBaseY, p.y);
    if (lipTopY <= noseBaseY + 1.0f) return box;
    const double cx = cxSum / upperLip.size();
    const double halfW = std::max(2.0, io * widthRatio);
    box.y0 = static_cast<int>(std::max(0.0, std::floor(noseBaseY - io * 0.02)));
    box.y1 = static_cast<int>(std::min(static_cast<double>(size.height), std::ceil(lipTopY + io * 0.04)));
    box.x0 = static_cast<int>(std::max(0.0, std::floor(cx - halfW)));
    box.x1 = static_cast<int>(std::min(static_cast<double>(size.width), std::ceil(cx + halfW)));
    box.valid = box.y1 > box.y0 && box.x1 > box.x0;
    return box;
}

// _carve_philtrum_from_protect()
cv::Mat carvePhiltrumFromProtect(const cv::Mat& protect, const Landmarks& lm, double io, double widthRatio) {
    const PhiltrumBox box = philtrumBox(protect.size(), lm, io, widthRatio);
    if (!box.valid) return protect;
    cv::Mat out = protect.clone();
    out(cv::Rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0)).setTo(0);
    return out;
}

// _build_philtrum_mask()
cv::Mat buildPhiltrumMask(cv::Size size, const Landmarks& lm, double io, const Section& params) {
    const double widthRatio = getF(params, {"philtrum_width_ratio"}, 0.16);
    const PhiltrumBox box = philtrumBox(size, lm, io, widthRatio * 1.08);
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    if (box.valid) mask(cv::Rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0)).setTo(255);
    return mask;
}

// _side_brow_top_1d()
std::vector<float> sideBrowTop1d(cv::Size size, const std::vector<cv::Point2f>& browPts, double io,
                                 double widthRatio, double yShiftRatio, double padUp) {
    const int w = size.width;
    std::vector<float> yOut(w, static_cast<float>(size.height));
    if (browPts.size() < 2) return yOut;
    const ArcCurve arc = smoothBrowArcCurve(size, browPts, io, widthRatio, yShiftRatio);
    for (int x = 0; x < w; ++x) {
        if (x >= arc.xLeft && x <= arc.xRight) {
            yOut[x] = arc.y[x] - static_cast<float>(padUp) - static_cast<float>(io * 0.003);
        }
    }
    return yOut;
}

// _forehead_brow_cutoff_y_map()
std::vector<float> foreheadBrowCutoffYMap(cv::Size size, const Landmarks& lm, double io, double yShiftRatio,
                                          double heightRatio, int dilatePx, int shrinkPx) {
    const auto [padUp, padDown] = browProtectThickness(io, dilatePx, shrinkPx, heightRatio);
    (void)padDown;
    std::vector<float> yCut(size.width, static_cast<float>(size.height));
    const std::pair<const std::vector<int>*, const std::vector<int>*> sides[] = {
        {&LEFT_BROW, &LEFT_EYE}, {&RIGHT_BROW, &RIGHT_EYE}};
    for (const auto& [browIdx, eyeIdx] : sides) {
        const auto brow = filterSideBrowPts(pickPts(lm, *browIdx), pickPts(lm, *eyeIdx), io);
        const auto top = sideBrowTop1d(size, brow, io, 1.0, yShiftRatio, padUp);
        for (int x = 0; x < size.width; ++x) yCut[x] = std::min(yCut[x], top[x]);
    }
    return yCut;
}

// _build_forehead_mask()
cv::Mat buildForeheadMask(cv::Size size, const Landmarks& lm, double io, const cv::Mat& faceMaskRaw,
                          const Section& params) {
    const double fhExpand = getF(params, {"forehead_height_ratio"}, 1.0);
    const auto browPts = vstack(pickPts(lm, LEFT_BROW), pickPts(lm, RIGHT_BROW));
    if (browPts.size() < 2) return cv::Mat::zeros(size, CV_8U);
    const double browBottomY = percentileOf(ys(browPts), 72, 0);
    const double liftPx = computeForeheadLiftPx(lm, io, pickPts(lm, FACE_OVAL), fhExpand);
    const cv::Mat oval = foreheadSupportOval(size, lm, faceMaskRaw, liftPx, browBottomY);

    const int browShrink = static_cast<int>(getF(params, {"brow_protect_shrink_radius"}, 2));
    const int protectRadius = static_cast<int>(getF(params, {"protect_expand_radius"}, 6));
    const double yShift = getF(params, {"brow_vertical_shift_ratio"}, 0.0);
    const double heightRatio = getF(params, {"brow_protect_height_ratio"}, 1.0);
    const auto yCut = foreheadBrowCutoffYMap(size, lm, io, yShift, heightRatio, protectRadius, browShrink);

    cv::Mat forehead = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < size.height; ++y) {
        const uchar* po = oval.ptr<uchar>(y);
        uchar* pf = forehead.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            if (po[x] > 0 && static_cast<float>(y) < yCut[x]) pf[x] = 255;
        }
    }

    if (fhExpand < 1.0 && countNonZero8(forehead) > 0) {
        int topY = size.height, bottomY = -1;
        for (int y = 0; y < size.height; ++y) {
            const uchar* pf = forehead.ptr<uchar>(y);
            for (int x = 0; x < size.width; ++x) {
                if (pf[x] > 0) {
                    topY = std::min(topY, y);
                    bottomY = std::max(bottomY, y);
                    break;
                }
            }
        }
        const int span = std::max(4, bottomY - topY);
        const int keepFrom = bottomY - static_cast<int>(span * std::max(0.35, fhExpand));
        for (int y = 0; y < std::min(keepFrom, size.height); ++y) forehead.row(y).setTo(0);
    }
    return forehead;
}

// _clip_to_skin() / _clip_forehead()
cv::Mat clipToSkin(const cv::Mat& mask, const cv::Mat& skin, const cv::Mat& protect) {
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* pm = mask.ptr<uchar>(y);
        const uchar* pk = skin.ptr<uchar>(y);
        const uchar* pp = protect.ptr<uchar>(y);
        uchar* po = out.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (pm[x] > 0 && pk[x] > 0 && pp[x] == 0) po[x] = 255;
        }
    }
    return out;
}

cv::Mat clipForehead(const cv::Mat& mask, const cv::Mat& protect) {
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* pm = mask.ptr<uchar>(y);
        const uchar* pp = protect.ptr<uchar>(y);
        uchar* po = out.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (pm[x] > 0 && pp[x] == 0) po[x] = 255;
        }
    }
    return out;
}

// _build_residual_brow_ridge_side()
cv::Mat buildResidualBrowRidgeSide(cv::Size size, const std::vector<cv::Point2f>& browPtsIn,
                                   const std::vector<cv::Point2f>& eyePts, double io, const cv::Mat& faceMask,
                                   const cv::Mat& skin, const cv::Mat& occupied, const cv::Mat& sideUMask,
                                   double heightRatio, double widthRatio, int dilatePx, int shrinkPx,
                                   double yShiftRatio) {
    const auto browPts = filterSideBrowPts(browPtsIn, eyePts, io);
    if (browPts.size() < 2 || eyePts.size() < 2) return cv::Mat::zeros(size, CV_8U);
    float ex0 = eyePts[0].x, ex1 = eyePts[0].x;
    for (const auto& p : eyePts) {
        ex0 = std::min(ex0, p.x);
        ex1 = std::max(ex1, p.x);
    }
    if (ex1 - ex0 < io * 0.07) return cv::Mat::zeros(size, CV_8U);

    const int h = size.height, w = size.width;
    const ArcCurve arc = smoothBrowArcCurve(size, browPts, io, widthRatio, yShiftRatio);
    const auto [padUp, padDown] = browProtectThickness(io, dilatePx, shrinkPx, heightRatio);
    (void)padUp;
    std::vector<float> yTop(w);
    for (int x = 0; x < w; ++x) yTop[x] = arc.y[x] + static_cast<float>(padDown);

    std::vector<cv::Point2f> eyeSorted = eyePts;
    std::stable_sort(eyeSorted.begin(), eyeSorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    std::vector<double> ex, ey;
    for (const auto& p : eyeSorted) {
        ex.push_back(p.x);
        ey.push_back(p.y);
    }
    std::vector<float> xx(w);
    for (int i = 0; i < w; ++i) xx[i] = static_cast<float>(i);
    std::vector<float> yEye = interp1d(xx, ex, ey);
    long long k = std::max<long long>(9, pyRound(io * 0.18));
    if (k % 2 == 0) ++k;
    cv::Mat row(1, w, CV_32F, yEye.data());
    cv::Mat blurred;
    cv::GaussianBlur(row, blurred, cv::Size(static_cast<int>(k), 1), 0);
    std::vector<float> yBot(w);
    for (int x = 0; x < w; ++x) yBot[x] = blurred.at<float>(0, x) + static_cast<float>(io * 0.018);

    cv::Mat ridge = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < h; ++y) {
        const uchar* pf = faceMask.ptr<uchar>(y);
        const uchar* pk = skin.ptr<uchar>(y);
        const uchar* pu = sideUMask.ptr<uchar>(y);
        const uchar* po = occupied.ptr<uchar>(y);
        uchar* pr = ridge.ptr<uchar>(y);
        const float yf = static_cast<float>(y);
        for (int x = 0; x < w; ++x) {
            if (pf[x] > 0 && pk[x] > 0 && pu[x] > 0 && x >= arc.xLeft && x <= arc.xRight && yf >= yTop[x] &&
                yf <= yBot[x] && po[x] == 0) {
                pr[x] = 255;
            }
        }
    }
    return ridge;
}

// _build_face_regions_simple()
RegionMasks buildFaceRegionsSimple(const cv::Mat& imageBgr, const Landmarks& lm, const Section& params) {
    const cv::Size size = imageBgr.size();
    RegionMasks out;
    out.geometry = getGeometry(lm, size);
    const double io = std::max(1.0f, out.geometry.interocular);

    const cv::Mat faceMaskRaw = buildLiftedFaceMaskRaw(size, lm, io, params);
    const int shrinkPx = static_cast<int>(std::max<long long>(0, pyRound(io * getF(params, {"face_contour_shrink_ratio"}, 0.02))));
    const cv::Mat faceMask = shrinkPx > 0 ? morphMask(faceMaskRaw, "erode", shrinkPx) : faceMaskRaw.clone();

    const int protectRadius = static_cast<int>(getF(params, {"protect_expand_radius"}, 6));
    const double eyeExtra = getF(params, {"eye_protect_extra_radius"}, 0.5);
    const int browShrink = static_cast<int>(getF(params, {"brow_protect_shrink_radius"}, 2));
    const double browHeightRatio = getF(params, {"brow_protect_height_ratio"}, 1.0);
    const double browWidthRatio = getF(params, {"brow_protect_width_ratio"}, 1.0);
    const double browYShift = getF(params, {"brow_vertical_shift_ratio"}, 0.12);
    const int lipShrink = static_cast<int>(getF(params, {"lip_protect_shrink_radius"}, 2));
    const int eyeRadius = std::max(1, protectRadius + static_cast<int>(pyRound(eyeExtra)));
    const int lipRadius = std::max(1, protectRadius - lipShrink);

    const cv::Mat leftEye = protectFeature(size, lm, LEFT_EYE, eyeRadius);
    const cv::Mat rightEye = protectFeature(size, lm, RIGHT_EYE, eyeRadius);
    std::vector<int> lipsIdx = OUTER_LIPS;
    lipsIdx.insert(lipsIdx.end(), INNER_LIPS.begin(), INNER_LIPS.end());
    const cv::Mat lips = protectFeature(size, lm, lipsIdx, lipRadius);
    const cv::Mat leftBrowProtect = buildBrowProtectMask(size, lm, LEFT_BROW, LEFT_EYE, io, protectRadius,
                                                         browShrink, browHeightRatio, browWidthRatio, browYShift);
    const cv::Mat rightBrowProtect = buildBrowProtectMask(size, lm, RIGHT_BROW, RIGHT_EYE, io, protectRadius,
                                                          browShrink, browHeightRatio, browWidthRatio, browYShift);
    cv::Mat protect = orMask(orMask(orMask(leftEye, rightEye), lips), orMask(leftBrowProtect, rightBrowProtect));
    cv::bitwise_and(protect, faceMaskRaw, protect);

    const int smoothRadius = static_cast<int>(getF(params, {"skin_mask_smooth_radius"}, 3));
    const cv::Mat skin = estimateSkinMask(imageBgr, faceMask, protect, smoothRadius);

    cv::Mat treatable = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < size.height; ++y) {
        const uchar* pf = faceMask.ptr<uchar>(y);
        const uchar* pk = skin.ptr<uchar>(y);
        const uchar* pp = protect.ptr<uchar>(y);
        uchar* pt = treatable.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            if (pf[x] > 0 && pk[x] > 0 && pp[x] == 0) pt[x] = 255;
        }
    }

    const cv::Mat empty = cv::Mat::zeros(size, CV_8U);
    if (countNonZero8(skin) == 0) out.warnings.push_back("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点");
    if (countNonZero8(treatable) == 0) out.warnings.push_back("可去高光皮肤区域为空，请检查人脸关键点");

    out.masks["face_mask_raw"] = faceMaskRaw;
    out.masks["face_mask"] = faceMask;
    out.masks["skin"] = skin;
    out.masks["protect"] = protect;
    out.masks["left_eye"] = leftEye;
    out.masks["right_eye"] = rightEye;
    out.masks["left_brow_protect"] = leftBrowProtect;
    out.masks["right_brow_protect"] = rightBrowProtect;
    out.masks["left_brow_ridge"] = empty;
    out.masks["right_brow_ridge"] = empty;
    out.masks["lips"] = lips;
    out.masks["mouth"] = lips;
    out.masks["nose_bridge"] = empty;
    out.masks["nose_tip"] = empty;
    out.masks["forehead"] = empty;
    out.masks["left_cheek"] = empty;
    out.masks["right_cheek"] = empty;
    out.masks["chin"] = empty;
    out.masks["philtrum"] = empty;
    out.masks["highlight_candidate"] = treatable;
    out.masks["treatable_skin"] = treatable;
    return out;
}

}  // namespace

cv::Mat estimateSkinMask(const cv::Mat& imageBgr, const cv::Mat& faceMask, const cv::Mat& protectMask, int smoothRadius) {
    const cv::Size size = imageBgr.size();
    cv::Mat hsv, ycrcb, lab;
    cv::cvtColor(imageBgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(imageBgr, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(imageBgr, lab, cv::COLOR_BGR2Lab);

    cv::Mat base(size, CV_8U);
    int baseCount = 0;
    for (int y = 0; y < size.height; ++y) {
        const uchar* pf = faceMask.ptr<uchar>(y);
        const uchar* pp = protectMask.ptr<uchar>(y);
        uchar* pb = base.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            pb[x] = (pf[x] > 0 && pp[x] == 0) ? 1 : 0;
            baseCount += pb[x];
        }
    }
    if (baseCount == 0) return cv::Mat::zeros(size, CV_8U);

    // 采样区：base & v>40 & v<245 & s>8，不足 50 个则退化为 base & v>25。
    std::vector<float> crS, cbS, aS, bS;
    for (int pass = 0; pass < 2; ++pass) {
        crS.clear();
        cbS.clear();
        aS.clear();
        bS.clear();
        for (int y = 0; y < size.height; ++y) {
            const uchar* pb = base.ptr<uchar>(y);
            const cv::Vec3b* ph = hsv.ptr<cv::Vec3b>(y);
            const cv::Vec3b* py = ycrcb.ptr<cv::Vec3b>(y);
            const cv::Vec3b* pl = lab.ptr<cv::Vec3b>(y);
            for (int x = 0; x < size.width; ++x) {
                if (!pb[x]) continue;
                const int v = ph[x][2], s = ph[x][1];
                const bool ok = (pass == 0) ? (v > 40 && v < 245 && s > 8) : (v > 25);
                if (!ok) continue;
                crS.push_back(py[x][1]);
                cbS.push_back(py[x][2]);
                aS.push_back(pl[x][1]);
                bS.push_back(pl[x][2]);
            }
        }
        if (crS.size() >= 50 || pass == 1) break;
    }
    const double medCr = medianOf(crS, 150);
    const double medCb = medianOf(cbS, 110);
    const double medA = medianOf(aS, 135);
    const double medB = medianOf(bS, 135);

    cv::Mat skinU8 = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < size.height; ++y) {
        const uchar* pb = base.ptr<uchar>(y);
        const cv::Vec3b* ph = hsv.ptr<cv::Vec3b>(y);
        const cv::Vec3b* py = ycrcb.ptr<cv::Vec3b>(y);
        const cv::Vec3b* pl = lab.ptr<cv::Vec3b>(y);
        uchar* pk = skinU8.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            if (!pb[x]) continue;
            const float v = ph[x][2], s = ph[x][1];
            const float cr = py[x][1], cb = py[x][2];
            const float a = pl[x][1], b = pl[x][2];
            const bool generic = cr >= 125 && cr <= 185 && cb >= 70 && cb <= 145 && v >= 35;
            const bool adaptive = std::abs(cr - medCr) <= 34 && std::abs(cb - medCb) <= 34 &&
                                  std::abs(a - medA) <= 28 && std::abs(b - medB) <= 34 && v >= 35;
            const bool brightHighlight = v >= 165 && s <= 175;
            const bool darkNonSkin = v < 45 || (v < 85 && s > 70);
            if ((generic || adaptive || brightHighlight) && !darkNonSkin) pk[x] = 255;
        }
    }
    skinU8 = morphMask(skinU8, "close", std::max(1, smoothRadius));
    skinU8 = morphMask(skinU8, "open", 1);
    if (smoothRadius > 0) {
        skinU8 = blurMask(skinU8, smoothRadius);
        for (int y = 0; y < size.height; ++y) {
            uchar* pk = skinU8.ptr<uchar>(y);
            for (int x = 0; x < size.width; ++x) pk[x] = pk[x] > 80 ? 255 : 0;
        }
    }
    for (int y = 0; y < size.height; ++y) {
        const uchar* pp = protectMask.ptr<uchar>(y);
        uchar* pk = skinU8.ptr<uchar>(y);
        for (int x = 0; x < size.width; ++x) {
            if (pp[x] > 0) pk[x] = 0;
        }
    }
    return skinU8;
}

RegionMasks buildFaceRegions(const cv::Mat& imageBgr, const Landmarks& lm, const Section& params) {
    // _use_simple_protect_mode()
    const bool keyRegionMode = getB(params, {"key_region_detection_mode"}, false);
    const bool simpleMode = !keyRegionMode && getB(params, {"simple_protect_mode"}, true);
    if (simpleMode) return buildFaceRegionsSimple(imageBgr, lm, params);

    const cv::Size size = imageBgr.size();
    const int h = size.height, w = size.width;
    RegionMasks out;
    out.geometry = getGeometry(lm, size);
    const FaceGeometry& geo = out.geometry;
    const double io = std::max(1.0f, geo.interocular);

    const cv::Mat faceMaskRaw = buildLiftedFaceMaskRaw(size, lm, io, params);
    const int shrinkPx = static_cast<int>(std::max<long long>(0, pyRound(io * getF(params, {"face_contour_shrink_ratio"}, 0.02))));
    const cv::Mat faceMask = shrinkPx > 0 ? morphMask(faceMaskRaw, "erode", shrinkPx) : faceMaskRaw.clone();

    const int protectRadius = static_cast<int>(getF(params, {"protect_expand_radius"}, 6));
    const double eyeExtra = getF(params, {"eye_protect_extra_radius"}, 0.5);
    const int browShrink = static_cast<int>(getF(params, {"brow_protect_shrink_radius"}, 2));
    const double browHeightRatio = getF(params, {"brow_protect_height_ratio"}, 1.0);
    const double browWidthRatio = getF(params, {"brow_protect_width_ratio"}, 1.0);
    const double browYShift = getF(params, {"brow_vertical_shift_ratio"}, 0.12);
    const double headYaw = estimateHeadYawRatio(lm);
    const int lipShrink = static_cast<int>(getF(params, {"lip_protect_shrink_radius"}, 2));
    const int eyeRadius = std::max(1, protectRadius + static_cast<int>(pyRound(eyeExtra)));
    const int lipRadius = std::max(1, protectRadius - lipShrink);

    const cv::Mat leftEye = protectFeature(size, lm, LEFT_EYE, eyeRadius);
    const cv::Mat rightEye = protectFeature(size, lm, RIGHT_EYE, eyeRadius);
    const cv::Mat leftBrowProtect = buildBrowProtectMask(size, lm, LEFT_BROW, LEFT_EYE, io, protectRadius,
                                                         browShrink, browHeightRatio, browWidthRatio, browYShift);
    const cv::Mat rightBrowProtect = buildBrowProtectMask(size, lm, RIGHT_BROW, RIGHT_EYE, io, protectRadius,
                                                          browShrink, browHeightRatio, browWidthRatio, browYShift);
    std::vector<int> lipsIdx = OUTER_LIPS;
    lipsIdx.insert(lipsIdx.end(), INNER_LIPS.begin(), INNER_LIPS.end());
    const cv::Mat lips = protectFeature(size, lm, lipsIdx, lipRadius);
    const cv::Mat browUnion = orMask(leftBrowProtect, rightBrowProtect);
    const bool protectBrows = getB(params, {"protect_brows"}, false);
    cv::Mat protect = orMask(orMask(leftEye, rightEye), lips);
    if (protectBrows) {
        protect = orMask(protect, browUnion);
    } else {
        cv::Mat notBrow;
        cv::bitwise_not(browUnion, notBrow);
        cv::bitwise_and(protect, notBrow, protect);
    }
    if (getB(params, {"philtrum_unprotect"}, true)) {
        protect = carvePhiltrumFromProtect(protect, lm, io, getF(params, {"philtrum_width_ratio"}, 0.16));
    }
    cv::bitwise_and(protect, faceMaskRaw, protect);

    cv::Mat skin = estimateSkinMask(imageBgr, faceMask, protect, static_cast<int>(getF(params, {"skin_mask_smooth_radius"}, 3)));

    // 鼻梁
    const double noseBridgeExpand = getF(params, {"nose_bridge_expand_ratio"}, 0.12);
    const int bridgeThickness = std::max(2, static_cast<int>(pyRound(io * noseBridgeExpand)));
    cv::Mat noseBridge = lineMask(size, pickPts(lm, NOSE_BRIDGE), bridgeThickness);
    noseBridge = morphMask(noseBridge, "dilate", std::max<long long>(1, pyRound(io * 0.015)));

    // 鼻尖
    const auto nosePts = pickPts(lm, NOSE_WING);
    double noseWidth, noseHeight;
    if (nosePts.size() >= 2) {
        float minX = nosePts[0].x, maxX = nosePts[0].x, minY = nosePts[0].y, maxY = nosePts[0].y;
        for (const auto& p : nosePts) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
        noseWidth = maxX - minX;
        noseHeight = maxY - minY;
    } else {
        noseWidth = io * 0.45;
        noseHeight = io * 0.35;
    }
    const double tipExpand = getF(params, {"nose_tip_expand_ratio"}, 0.72);
    const cv::Point2f tipCenter = lm.size() > 4 ? cv::Point2f(lm[4][0], lm[4][1]) : geo.nose_tip;
    cv::Mat noseTip = ellipseMask(size, tipCenter,
                                  cv::Size2f(static_cast<float>(std::max(2.0, noseWidth * tipExpand * 0.55)),
                                             static_cast<float>(std::max(2.0, noseHeight * tipExpand * 0.75))),
                                  geo.angle_deg);

    cv::Mat forehead = buildForeheadMask(size, lm, io, faceMaskRaw, params);
    forehead = clipForehead(forehead, protect);
    cv::Mat philtrum = buildPhiltrumMask(size, lm, io, params);

    // 姿态自适应局部坐标
    const cv::Point2f origin = geo.nose_tip;
    const auto ovalP = pickPts(lm, FACE_OVAL);
    std::vector<float> ovalU, ovalV;
    projectPoints(ovalP, origin, geo.x_axis, geo.y_axis, ovalU, ovalV);
    double faceWLocal = io * 2.0, faceHLocal = io * 2.8;
    if (!ovalU.empty()) {
        const auto [minU, maxU] = std::minmax_element(ovalU.begin(), ovalU.end());
        const auto [minV, maxV] = std::minmax_element(ovalV.begin(), ovalV.end());
        faceWLocal = std::max(1.0f, *maxU - *minU);
        faceHLocal = std::max(1.0f, *maxV - *minV);
    }

    auto lowerEyeP = vstack(pickPts(lm, LEFT_LOWER_EYE), pickPts(lm, RIGHT_LOWER_EYE));
    std::vector<float> tmpU, lowerEyeV;
    projectPoints(lowerEyeP, origin, geo.x_axis, geo.y_axis, tmpU, lowerEyeV);
    std::vector<int> mouthIdx = OUTER_LIPS;
    mouthIdx.insert(mouthIdx.end(), MOUTH_CORNERS.begin(), MOUTH_CORNERS.end());
    std::vector<float> mouthV;
    projectPoints(pickPts(lm, mouthIdx), origin, geo.x_axis, geo.y_axis, tmpU, mouthV);
    std::vector<float> noseU, tmpV;
    projectPoints(pickPts(lm, NOSE_WING), origin, geo.x_axis, geo.y_axis, noseU, tmpV);
    std::vector<int> chinIdx = CHIN_AREA;
    chinIdx.push_back(152);
    std::vector<float> chinV;
    projectPoints(pickPts(lm, chinIdx), origin, geo.x_axis, geo.y_axis, tmpU, chinV);

    const double cheekUpper = !lowerEyeV.empty() ? medianOf(lowerEyeV, 0) - 0.05 * faceHLocal : -0.15 * faceHLocal;
    const double cheekLower = !mouthV.empty() ? percentileOf(mouthV, 75, 0) + 0.07 * faceHLocal : 0.45 * faceHLocal;
    const double noseHalf = !noseU.empty()
        ? std::max(io * 0.08, (percentileOf(noseU, 85, 0) - percentileOf(noseU, 15, 0)) / 2.0)
        : io * 0.12;
    const double cheekExpand = getF(params, {"cheek_expand_ratio"}, 1.0);
    const double cheekMargin = (cheekExpand - 1.0) * 0.06 * faceHLocal;
    const double midlineU = 0.0;

    const double lowerLipV = !mouthV.empty() ? percentileOf(mouthV, 90, 0) : 0.35 * faceHLocal;
    const double chinUpper = lowerLipV - 0.01 * faceHLocal;
    const double chinLower = !chinV.empty() ? *std::max_element(chinV.begin(), chinV.end()) + 0.02 * faceHLocal
                                            : 0.70 * faceHLocal;

    cv::Mat leftCheek = cv::Mat::zeros(size, CV_8U);
    cv::Mat rightCheek = cv::Mat::zeros(size, CV_8U);
    cv::Mat chin = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < h; ++y) {
        const uchar* pf = faceMask.ptr<uchar>(y);
        uchar* plc = leftCheek.ptr<uchar>(y);
        uchar* prc = rightCheek.ptr<uchar>(y);
        uchar* pc = chin.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const float dx = static_cast<float>(x) - origin.x;
            const float dy = static_cast<float>(y) - origin.y;
            const float u = dx * geo.x_axis[0] + dy * geo.x_axis[1];
            const float v = dx * geo.y_axis[0] + dy * geo.y_axis[1];
            const bool band = v >= cheekUpper - cheekMargin && v <= cheekLower + cheekMargin;
            if (pf[x] > 0) {
                if (u < midlineU - noseHalf * 0.75 && band) plc[x] = 255;
                if (u > midlineU + noseHalf * 0.75 && band) prc[x] = 255;
                if (v >= chinUpper && v <= chinLower) pc[x] = 255;
            }
        }
    }
    const double chinExpand = getF(params, {"chin_expand_ratio"}, 1.0);
    if (chinExpand > 1.01) chin = morphMask(chin, "dilate", pyRound(io * 0.04 * (chinExpand - 1.0)));

    noseBridge = clipToSkin(noseBridge, skin, protect);
    noseTip = clipToSkin(noseTip, skin, protect);
    chin = clipToSkin(chin, skin, protect);
    leftCheek = clipToSkin(leftCheek, skin, protect);
    rightCheek = clipToSkin(rightCheek, skin, protect);
    philtrum = clipForehead(philtrum, protect);

    // 眉弓 = 剩余集合
    cv::Mat priority = orMask(forehead, orMask(leftCheek, rightCheek));
    priority = orMask(priority, orMask(leftEye, rightEye));
    priority = orMask(priority, orMask(leftBrowProtect, rightBrowProtect));
    priority = orMask(priority, orMask(noseBridge, noseTip));

    cv::Mat leftSide = cv::Mat::zeros(size, CV_8U);
    cv::Mat rightSide = cv::Mat::zeros(size, CV_8U);
    if (std::abs(headYaw) > 0.12) {
        const double eyeMidX = geo.eye_mid.x;
        for (int x = 0; x < w; ++x) {
            if (x < eyeMidX - io * 0.04) leftSide.col(x).setTo(255);
            if (x > eyeMidX + io * 0.04) rightSide.col(x).setTo(255);
        }
    } else {
        for (int y = 0; y < h; ++y) {
            uchar* pl = leftSide.ptr<uchar>(y);
            uchar* pr = rightSide.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                const float dx = static_cast<float>(x) - origin.x;
                const float dy = static_cast<float>(y) - origin.y;
                const float u = dx * geo.x_axis[0] + dy * geo.x_axis[1];
                if (u < midlineU - noseHalf * 0.35) pl[x] = 255;
                if (u > midlineU + noseHalf * 0.35) pr[x] = 255;
            }
        }
    }
    const cv::Mat leftBrowRidge = buildResidualBrowRidgeSide(
        size, pickPts(lm, LEFT_BROW), pickPts(lm, LEFT_EYE), io, faceMask, skin, priority, leftSide,
        browHeightRatio, browWidthRatio, protectRadius, browShrink, browYShift);
    const cv::Mat rightBrowRidge = buildResidualBrowRidgeSide(
        size, pickPts(lm, RIGHT_BROW), pickPts(lm, RIGHT_EYE), io, faceMask, skin, priority, rightSide,
        browHeightRatio, browWidthRatio, protectRadius, browShrink, browYShift);
    const cv::Mat browRidge = orMask(leftBrowRidge, rightBrowRidge);
    {
        cv::Mat notRidge;
        cv::bitwise_not(browRidge, notRidge);
        cv::bitwise_and(protect, notRidge, protect);
        cv::bitwise_and(forehead, notRidge, forehead);
    }

    // 皮肤必须覆盖额头候选区
    skin = orMask(skin, forehead);

    // 互斥清理
    cv::Mat nonCheek = orMask(orMask(noseBridge, noseTip), orMask(forehead, chin));
    nonCheek = orMask(nonCheek, philtrum);
    nonCheek = orMask(nonCheek, browRidge);
    nonCheek = orMask(nonCheek, protect);
    for (int y = 0; y < h; ++y) {
        const uchar* pn = nonCheek.ptr<uchar>(y);
        uchar* plc = leftCheek.ptr<uchar>(y);
        uchar* prc = rightCheek.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (pn[x] > 0) {
                plc[x] = 0;
                prc[x] = 0;
            }
        }
    }

    cv::Mat candidate = cv::Mat::zeros(size, CV_8U);
    const std::vector<const cv::Mat*> regionParts = {&noseBridge, &noseTip,    &forehead, &leftCheek,
                                                     &rightCheek, &chin,       &philtrum, &leftBrowRidge,
                                                     &rightBrowRidge};
    for (const cv::Mat* m : regionParts) {
        candidate = orMask(candidate, *m);
    }

    if (countNonZero8(skin) == 0) out.warnings.push_back("皮肤区域生成失败，请调低皮肤 mask 平滑半径或检查关键点");
    if (countNonZero8(candidate) == 0) out.warnings.push_back("高光候选区域为空，请检查区域扩张参数");

    out.masks["face_mask_raw"] = faceMaskRaw;
    out.masks["face_mask"] = faceMask;
    out.masks["skin"] = skin;
    out.masks["protect"] = protect;
    out.masks["left_eye"] = leftEye;
    out.masks["right_eye"] = rightEye;
    out.masks["left_brow_protect"] = leftBrowProtect;
    out.masks["right_brow_protect"] = rightBrowProtect;
    out.masks["left_brow_ridge"] = leftBrowRidge;
    out.masks["right_brow_ridge"] = rightBrowRidge;
    out.masks["lips"] = lips;
    out.masks["mouth"] = lips;
    out.masks["nose_bridge"] = noseBridge;
    out.masks["nose_tip"] = noseTip;
    out.masks["forehead"] = forehead;
    out.masks["left_cheek"] = leftCheek;
    out.masks["right_cheek"] = rightCheek;
    out.masks["chin"] = chin;
    out.masks["philtrum"] = philtrum;
    out.masks["highlight_candidate"] = candidate;
    return out;
}

}  // namespace hr
