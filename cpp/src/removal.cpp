#include "high_removal/removal.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

namespace hr {

namespace {

// _edge_protected_alpha()：返回 [0,1] 的 float 掩码。
cv::Mat1f edgeProtectedAlpha(const cv::Mat& imageBgr, const cv::Mat& softMask, double edgeStrength) {
    cv::Mat1f alpha(imageBgr.size());
    for (int y = 0; y < alpha.rows; ++y) {
        const uchar* ps = softMask.ptr<uchar>(y);
        float* pa = alpha[y];
        for (int x = 0; x < alpha.cols; ++x) pa[x] = static_cast<float>(ps[x]) / 255.f;
    }
    if (edgeStrength <= 0) return alpha;
    cv::Mat gray, edges;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 60, 130);
    cv::GaussianBlur(edges, edges, cv::Size(5, 5), 0);
    const float strength = static_cast<float>(std::max(0.0, std::min(1.0, edgeStrength)));
    for (int y = 0; y < alpha.rows; ++y) {
        const uchar* pe = edges.ptr<uchar>(y);
        float* pa = alpha[y];
        for (int x = 0; x < alpha.cols; ++x) {
            const float e = static_cast<float>(pe[x]) / 255.f;
            pa[x] = std::max(0.f, std::min(1.f, pa[x] * (1.f - strength * e)));
        }
    }
    return alpha;
}

// _inpaint_lab_reference()：返回 CV_32FC3 的 LAB 参考。
cv::Mat inpaintLabReference(const cv::Mat& imageBgr, const cv::Mat& hardMask, int radius) {
    radius = std::max(1, std::min(9, radius));
    cv::Mat lab;
    cv::cvtColor(imageBgr, lab, cv::COLOR_BGR2Lab);
    cv::Mat labF;
    if (cv::countNonZero(hardMask) == 0) {
        lab.convertTo(labF, CV_32FC3);
        return labF;
    }
    // 分通道 TELEA inpaint。
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    for (auto& c : channels) {
        cv::Mat filled;
        cv::inpaint(c, hardMask, filled, radius, cv::INPAINT_TELEA);
        c = filled;
    }
    cv::Mat ref;
    cv::merge(channels, ref);
    cv::Mat refF;
    ref.convertTo(refF, CV_32FC3);
    // 双边滤波肤色基线。
    cv::Mat bilateral;
    cv::bilateralFilter(imageBgr, bilateral, 11, 45, 45);
    cv::Mat ref2;
    cv::cvtColor(bilateral, ref2, cv::COLOR_BGR2Lab);
    cv::Mat ref2F;
    ref2.convertTo(ref2F, CV_32FC3);
    cv::Mat out(imageBgr.size(), CV_32FC3);
    for (int y = 0; y < out.rows; ++y) {
        const cv::Vec3f* p1 = refF.ptr<cv::Vec3f>(y);
        const cv::Vec3f* p2 = ref2F.ptr<cv::Vec3f>(y);
        cv::Vec3f* po = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out.cols; ++x) {
            po[x] = 0.72f * p1[x] + 0.28f * p2[x];
        }
    }
    return out;
}

// _faithful_suppress()
cv::Mat faithfulSuppress(const cv::Mat& imageBgr, const cv::Mat& hardMask, const cv::Mat& softMask,
                         const Section& params) {
    const float brightness = static_cast<float>(getF(params, {"brightness_suppress_strength"}, 0.74));
    const float colorStrength = static_cast<float>(getF(params, {"chroma_restore_strength"}, 0.30));
    const float textureStrength = static_cast<float>(getF(params, {"texture_preserve_strength"}, 0.78));
    const double edgeStrength = getF(params, {"edge_protect_strength"}, 0.45);
    const float finalAlpha = static_cast<float>(getF(params, {"final_blend_alpha"}, 0.92));
    const int radius = static_cast<int>(getF(params, {"inpainting_radius"}, 3));
    const float luminanceFloor = static_cast<float>(getF(params, {"faithful_luminance_floor"}, 0.86));

    const cv::Mat1f alpha0 = edgeProtectedAlpha(imageBgr, softMask, edgeStrength);

    cv::Mat lab8;
    cv::cvtColor(imageBgr, lab8, cv::COLOR_BGR2Lab);
    cv::Mat lab;
    lab8.convertTo(lab, CV_32FC3);
    const cv::Mat refLab = inpaintLabReference(imageBgr, hardMask, radius);

    cv::Mat1f L(imageBgr.size());
    for (int y = 0; y < L.rows; ++y) {
        const cv::Vec3f* pl = lab.ptr<cv::Vec3f>(y);
        for (int x = 0; x < L.cols; ++x) L(y, x) = pl[x][0];
    }
    cv::Mat1f low;
    cv::GaussianBlur(L, low, cv::Size(0, 0), 2.2);
    cv::Mat1f localSkin;
    cv::GaussianBlur(L, localSkin, cv::Size(0, 0), std::max(8.0, getF(params, {"local_sigma"}, 6.0) * 1.6));

    const float textureCoef = 0.16f * std::max(0.f, std::min(1.f, textureStrength));
    cv::Mat outLab(imageBgr.size(), CV_8UC3);
    for (int y = 0; y < L.rows; ++y) {
        const cv::Vec3f* plab = lab.ptr<cv::Vec3f>(y);
        const cv::Vec3f* pref = refLab.ptr<cv::Vec3f>(y);
        const float* pa0 = alpha0[y];
        cv::Vec3b* po = outLab.ptr<cv::Vec3b>(y);
        for (int x = 0; x < L.cols; ++x) {
            const float aL = std::max(0.f, std::min(1.f, pa0[x] * brightness * finalAlpha));
            const float aC = std::max(0.f, std::min(1.f, pa0[x] * colorStrength * finalAlpha));
            const float Lv = plab[x][0];
            const float refL = pref[x][0];
            const float texture = Lv - low(y, x);
            const float textureKeep = texture * textureCoef;
            const float ls = localSkin(y, x);
            const float minAllowed = std::max(Lv * luminanceFloor, ls * 0.93f);
            float targetL = std::max(refL + textureKeep, minAllowed);
            targetL = std::max(targetL, ls - 5.f);
            targetL = std::min(targetL, Lv + 0.5f);
            const float outL = Lv * (1.f - aL) + targetL * aL;
            const float outA = plab[x][1] * (1.f - aC) + pref[x][1] * aC;
            const float outB = plab[x][2] * (1.f - aC) + pref[x][2] * aC;
            // np.clip(...,0,255).astype(np.uint8)：截断
            po[x] = cv::Vec3b(truncU8(std::max(0.f, std::min(255.f, outL))),
                              truncU8(std::max(0.f, std::min(255.f, outA))),
                              truncU8(std::max(0.f, std::min(255.f, outB))));
        }
    }
    cv::Mat outBgr;
    cv::cvtColor(outLab, outBgr, cv::COLOR_Lab2BGR);
    // keep = alpha0 <= 0.001 处回写原图
    for (int y = 0; y < outBgr.rows; ++y) {
        const float* pa0 = alpha0[y];
        const cv::Vec3b* pi = imageBgr.ptr<cv::Vec3b>(y);
        cv::Vec3b* po = outBgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < outBgr.cols; ++x) {
            if (pa0[x] <= 0.001f) po[x] = pi[x];
        }
    }
    return outBgr;
}

// _strong_inpaint()
cv::Mat strongInpaint(const cv::Mat& imageBgr, const cv::Mat& hardMask, const cv::Mat& softMask,
                      const Section& params, const cv::Mat* baseImage) {
    const cv::Mat& source = baseImage ? *baseImage : imageBgr;
    const int radius = std::max(1, std::min(9, static_cast<int>(getF(params, {"inpainting_radius"}, 3))));
    const float alphaStrength = static_cast<float>(getF(params, {"poisson_alpha_strength"}, 0.42));
    const float finalAlpha = static_cast<float>(getF(params, {"final_blend_alpha"}, 0.92));
    if (cv::countNonZero(hardMask) == 0) return source.clone();
    cv::Mat inpainted;
    cv::inpaint(source, hardMask, inpainted, radius, cv::INPAINT_TELEA);
    cv::Mat out(source.size(), CV_8UC3);
    for (int y = 0; y < out.rows; ++y) {
        const uchar* ps = softMask.ptr<uchar>(y);
        const cv::Vec3b* psrc = source.ptr<cv::Vec3b>(y);
        const cv::Vec3b* pin = inpainted.ptr<cv::Vec3b>(y);
        cv::Vec3b* po = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out.cols; ++x) {
            const float alpha = std::max(0.f, std::min(1.f, ps[x] / 255.f * alphaStrength * finalAlpha));
            for (int c = 0; c < 3; ++c) {
                const float v = psrc[x][c] * (1.f - alpha) + pin[x][c] * alpha;
                po[x][c] = truncU8(std::max(0.f, std::min(255.f, v)));
            }
        }
    }
    return out;
}

// _remove_highlight_core()
RemovalOutput removeHighlightCore(const cv::Mat& imageBgr, const cv::Mat& hardMask, const cv::Mat& softMask,
                                  const RegionMasks& regions, const Section& params) {
    RemovalOutput out;
    const std::string mode = getS(params, {"mode"}, "保真");
    const cv::Mat& skinM = regions.at("skin");
    const cv::Mat& protectM = regions.at("protect");

    cv::Mat hard = hardMask.clone();
    cv::Mat soft = softMask.clone();
    for (int y = 0; y < hard.rows; ++y) {
        const uchar* pk = skinM.ptr<uchar>(y);
        const uchar* pp = protectM.ptr<uchar>(y);
        uchar* ph = hard.ptr<uchar>(y);
        uchar* ps = soft.ptr<uchar>(y);
        for (int x = 0; x < hard.cols; ++x) {
            if (pk[x] == 0 || pp[x] > 0) {
                ph[x] = 0;
                ps[x] = 0;
            }
        }
    }

    const int faceArea = std::max(1, cv::countNonZero(regions.at("face_mask")));
    const double modAreaRatio = static_cast<double>(cv::countNonZero(hard)) / faceArea;
    const double maxAllowed = getF(params, {"max_allowed_modify_area_ratio"}, 0.15);
    if (modAreaRatio > maxAllowed) {
        out.warnings.push_back("当前修改区域过大，可能影响真实性，请降低处理强度");
    }

    cv::Mat result;
    if (mode == "强修复") {
        result = strongInpaint(imageBgr, hard, soft, params, nullptr);
    } else if (mode == "混合") {
        const cv::Mat faithful = faithfulSuppress(imageBgr, hard, soft, params);
        cv::Mat lab;
        cv::cvtColor(imageBgr, lab, cv::COLOR_BGR2Lab);
        const float extra = static_cast<float>(getF(params, {"extreme_core_extra_l"}, 12));
        const float labLThr = static_cast<float>(getF(params, {"lab_l_threshold"}, 188));
        cv::Mat extreme = cv::Mat::zeros(imageBgr.size(), CV_8U);
        int extremeCount = 0;
        for (int y = 0; y < extreme.rows; ++y) {
            const uchar* ph = hard.ptr<uchar>(y);
            const cv::Vec3b* pl = lab.ptr<cv::Vec3b>(y);
            uchar* pe = extreme.ptr<uchar>(y);
            for (int x = 0; x < extreme.cols; ++x) {
                if (ph[x] > 0 && static_cast<float>(pl[x][0]) >= labLThr + extra) {
                    pe[x] = 255;
                    ++extremeCount;
                }
            }
        }
        if (extremeCount > 0) {
            cv::Mat extremeSoft;
            const double sigma = std::max(1.0, getF(params, {"mask_blur_radius"}, 9) / 2.5);
            cv::GaussianBlur(extreme, extremeSoft, cv::Size(0, 0), sigma);
            result = strongInpaint(imageBgr, extreme, extremeSoft, params, &faithful);
        } else {
            result = faithful;
        }
    } else {
        result = faithfulSuppress(imageBgr, hard, soft, params);
    }

    // 强制保护：五官、背景、非皮肤区域完全回写原图。
    for (int y = 0; y < result.rows; ++y) {
        const uchar* ps = soft.ptr<uchar>(y);
        const uchar* pp = protectM.ptr<uchar>(y);
        const uchar* pk = skinM.ptr<uchar>(y);
        const cv::Vec3b* pi = imageBgr.ptr<cv::Vec3b>(y);
        cv::Vec3b* pr = result.ptr<cv::Vec3b>(y);
        for (int x = 0; x < result.cols; ++x) {
            if (ps[x] == 0 || pp[x] > 0 || pk[x] == 0) pr[x] = pi[x];
        }
    }
    out.result_bgr = result;
    return out;
}

}  // namespace

RemovalOutput removeHighlight(const cv::Mat& imageBgr, const cv::Mat& hardMask, const cv::Mat& softMask,
                              const RegionMasks& regions, const Section& params) {
    const bool enableRoi = getB(params, {"enable_roi_removal"}, true);
    const double paddingRatio = getF(params, {"roi_padding_ratio"}, 0.12);
    if (enableRoi) {
        const auto roi = computeRoiBbox(hardMask, softMask, imageBgr.size(), paddingRatio);
        if (roi.has_value()) {
            const auto [x, y, rw, rh] = *roi;
            if (rw < imageBgr.cols || rh < imageBgr.rows) {
                const cv::Rect rect(x, y, rw, rh);
                RegionMasks cropped;
                cropped.geometry = regions.geometry;
                cropped.warnings = regions.warnings;
                for (const auto& [key, mask] : regions.masks) {
                    if (mask.dims == 2 && mask.size() == imageBgr.size()) {
                        cropped.masks[key] = mask(rect);
                    }
                }
                RemovalOutput cropOut = removeHighlightCore(imageBgr(rect), hardMask(rect), softMask(rect), cropped, params);
                RemovalOutput out;
                out.warnings = cropOut.warnings;
                out.result_bgr = imageBgr.clone();
                cropOut.result_bgr.copyTo(out.result_bgr(rect));
                return out;
            }
        }
    }
    return removeHighlightCore(imageBgr, hardMask, softMask, regions, params);
}

}  // namespace hr
