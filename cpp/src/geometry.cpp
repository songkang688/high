#include "high_removal/geometry.hpp"

#include <cmath>

#include "high_removal/topology.hpp"

namespace hr {

std::vector<cv::Point2f> pickPts(const Landmarks& lm, const std::vector<int>& indices) {
    std::vector<cv::Point2f> out;
    out.reserve(indices.size());
    const int n = static_cast<int>(lm.size());
    for (int idx : indices) {
        if (idx < 0 || idx >= n) continue;
        const auto& p = lm[idx];
        if (!std::isfinite(p[0]) || !std::isfinite(p[1])) continue;
        out.emplace_back(p[0], p[1]);
    }
    return out;
}

cv::Point2f centerOf(const Landmarks& lm, const std::vector<int>& indices) {
    const auto p = pickPts(lm, indices);
    if (p.empty()) return {0.f, 0.f};
    double sx = 0, sy = 0;
    for (const auto& q : p) {
        sx += q.x;
        sy += q.y;
    }
    return {static_cast<float>(sx / p.size()), static_cast<float>(sy / p.size())};
}

Bbox bboxFromLandmarks(const Landmarks& lm, cv::Size imageSize) {
    auto p = pickPts(lm, topo::FACE_OVAL);
    if (p.empty()) {
        p.reserve(lm.size());
        for (const auto& q : lm) p.emplace_back(q[0], q[1]);
    }
    float minX = p[0].x, minY = p[0].y, maxX = p[0].x, maxY = p[0].y;
    for (const auto& q : p) {
        minX = std::min(minX, q.x);
        minY = std::min(minY, q.y);
        maxX = std::max(maxX, q.x);
        maxY = std::max(maxY, q.y);
    }
    const int w = imageSize.width, h = imageSize.height;
    auto clampI = [](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };
    const int x0 = clampI(static_cast<int>(std::floor(minX)), 0, w - 1);
    const int y0 = clampI(static_cast<int>(std::floor(minY)), 0, h - 1);
    const int x1 = clampI(static_cast<int>(std::ceil(maxX)), 0, w - 1);
    const int y1 = clampI(static_cast<int>(std::ceil(maxY)), 0, h - 1);
    return Bbox{x0, y0, std::max(0, x1 - x0 + 1), std::max(0, y1 - y0 + 1)};
}

FaceGeometry getGeometry(const Landmarks& lm, cv::Size imageSize) {
    FaceGeometry g;
    const cv::Point2f le = centerOf(lm, topo::LEFT_EYE);
    const cv::Point2f re = centerOf(lm, topo::RIGHT_EYE);
    const cv::Point2f eyeMid((le.x + re.x) / 2.f, (le.y + re.y) / 2.f);
    const cv::Point2f nose = centerOf(lm, {4});
    const cv::Point2f mouthL = centerOf(lm, {61});
    const cv::Point2f mouthR = centerOf(lm, {291});
    const cv::Point2f chin = centerOf(lm, {152});
    cv::Vec2f vec(re.x - le.x, re.y - le.y);
    float io = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    if (io < 1.f) {
        io = static_cast<float>(std::max(imageSize.width, imageSize.height) * 0.1);
        vec = {1.f, 0.f};
    }
    const float norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-6f;
    cv::Vec2f xAxis(vec[0] / norm, vec[1] / norm);
    cv::Vec2f yAxis(-xAxis[1], xAxis[0]);
    const float dot = (chin.x - eyeMid.x) * yAxis[0] + (chin.y - eyeMid.y) * yAxis[1];
    if (dot < 0) yAxis = -yAxis;
    g.left_eye_center = le;
    g.right_eye_center = re;
    g.eye_mid = eyeMid;
    g.nose_tip = nose;
    g.mouth_left = mouthL;
    g.mouth_right = mouthR;
    g.chin = chin;
    g.interocular = io;
    g.x_axis = xAxis;
    g.y_axis = yAxis;
    g.angle_deg = static_cast<float>(std::atan2(xAxis[1], xAxis[0]) * 180.0 / CV_PI);
    g.face_bbox = bboxFromLandmarks(lm, imageSize);
    return g;
}

double estimateHeadYawRatio(const Landmarks& lm) {
    const cv::Point2f le = centerOf(lm, topo::LEFT_EYE);
    const cv::Point2f re = centerOf(lm, topo::RIGHT_EYE);
    const cv::Point2f nose = centerOf(lm, {4});
    const cv::Vec2f vec(re.x - le.x, re.y - le.y);
    const double io = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    if (io < 1.0) return 0.0;
    const cv::Vec2f xAxis(static_cast<float>(vec[0] / io), static_cast<float>(vec[1] / io));
    const cv::Point2f eyeMid((le.x + re.x) * 0.5f, (le.y + re.y) * 0.5f);
    return ((nose.x - eyeMid.x) * xAxis[0] + (nose.y - eyeMid.y) * xAxis[1]) / io;
}

std::vector<cv::Point2f> filterSideBrowPts(const std::vector<cv::Point2f>& brow, const std::vector<cv::Point2f>& eye, double io) {
    if (brow.size() < 2 || eye.size() < 2) return brow;
    float ex0 = eye[0].x, ex1 = eye[0].x, eyTop = eye[0].y;
    for (const auto& p : eye) {
        ex0 = std::min(ex0, p.x);
        ex1 = std::max(ex1, p.x);
        eyTop = std::min(eyTop, p.y);
    }
    std::vector<cv::Point2f> filtered;
    for (const auto& p : brow) {
        if (p.x >= ex0 - io * 0.14 && p.x <= ex1 + io * 0.14 && p.y <= eyTop + io * 0.06) {
            filtered.push_back(p);
        }
    }
    if (filtered.size() >= 2) {
        float minX = filtered[0].x, maxX = filtered[0].x;
        for (const auto& p : filtered) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
        }
        if (maxX - minX >= io * 0.05) return filtered;
    }
    return brow;
}

void projectPoints(const std::vector<cv::Point2f>& p, cv::Point2f origin, cv::Vec2f xAxis, cv::Vec2f yAxis,
                   std::vector<float>& u, std::vector<float>& v) {
    u.resize(p.size());
    v.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        const float dx = p[i].x - origin.x;
        const float dy = p[i].y - origin.y;
        u[i] = dx * xAxis[0] + dy * xAxis[1];
        v[i] = dx * yAxis[0] + dy * yAxis[1];
    }
}

}  // namespace hr
