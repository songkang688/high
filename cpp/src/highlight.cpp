#include "high_removal/highlight.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace hr {

namespace {

// _masked_gaussian()
cv::Mat1f maskedGaussian(const cv::Mat1f& values, const cv::Mat1b& valid, double sigma) {
    cv::Mat1f validF(values.size());
    cv::Mat1f prod(values.size());
    for (int y = 0; y < values.rows; ++y) {
        const float* pv = values[y];
        const uchar* pb = valid[y];
        float* pf = validF[y];
        float* pp = prod[y];
        for (int x = 0; x < values.cols; ++x) {
            pf[x] = pb[x] ? 1.f : 0.f;
            pp[x] = pv[x] * pf[x];
        }
    }
    cv::Mat1f num, den;
    cv::GaussianBlur(prod, num, cv::Size(0, 0), sigma, sigma);
    cv::GaussianBlur(validF, den, cv::Size(0, 0), sigma, sigma);
    cv::Mat1f out(values.size());
    for (int y = 0; y < values.rows; ++y) {
        const float* pn = num[y];
        const float* pd = den[y];
        float* po = out[y];
        for (int x = 0; x < values.cols; ++x) po[x] = pn[x] / std::max(pd[x], 1e-3f);
    }
    return out;
}

// _grow_from_core()
cv::Mat1b growFromCore(const cv::Mat1b& core, const cv::Mat1b& halo, int maxIter) {
    cv::Mat1b grown = core.clone();
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    for (int i = 0; i < std::max(0, maxIter); ++i) {
        cv::Mat1b dilated;
        cv::dilate(grown, dilated, kernel);
        cv::Mat1b nxt(grown.size());
        bool changed = false;
        for (int y = 0; y < grown.rows; ++y) {
            const uchar* pd = dilated[y];
            const uchar* ph = halo[y];
            const uchar* pg = grown[y];
            uchar* pn = nxt[y];
            for (int x = 0; x < grown.cols; ++x) {
                pn[x] = (pd[x] && ph[x]) ? 255 : 0;
                changed |= (pn[x] != 0) != (pg[x] != 0);
            }
        }
        if (!changed) break;
        grown = nxt;
    }
    return grown;
}

// _cap_by_score()
cv::Mat1b capByScore(const cv::Mat1b& mask, const cv::Mat1f& score, const cv::Mat1b& region, int maxPixels) {
    cv::Mat1b out(mask.size());
    int count = 0;
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* pm = mask[y];
        const uchar* pr = region[y];
        uchar* po = out[y];
        for (int x = 0; x < mask.cols; ++x) {
            po[x] = (pm[x] && pr[x]) ? 255 : 0;
            count += po[x] ? 1 : 0;
        }
    }
    if (maxPixels <= 0 || count <= maxPixels) return out;
    std::vector<float> vals;
    vals.reserve(count);
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* po = out[y];
        const float* ps = score[y];
        for (int x = 0; x < mask.cols; ++x) {
            if (po[x]) vals.push_back(ps[x]);
        }
    }
    if (static_cast<int>(vals.size()) <= maxPixels) return out;
    const float kth = kthLargest(vals, maxPixels);
    for (int y = 0; y < mask.rows; ++y) {
        uchar* po = out[y];
        const float* ps = score[y];
        for (int x = 0; x < mask.cols; ++x) {
            if (po[x] && !(ps[x] >= kth)) po[x] = 0;
        }
    }
    return out;
}

// _region_fraction()
double regionFraction(const std::string& key, const Section& params) {
    if (key == "forehead") return getF(params, {"forehead_region_max_fraction"}, 0.30);
    if (key == "nose_bridge") return getF(params, {"nose_bridge_region_max_fraction"}, 0.72);
    if (key == "nose_tip") return getF(params, {"nose_tip_region_max_fraction"}, 0.55);
    if (key == "left_cheek" || key == "right_cheek") return getF(params, {"cheek_region_max_fraction"}, 0.20);
    if (key == "chin") return getF(params, {"chin_region_max_fraction"}, 0.24);
    if (key == "philtrum") return getF(params, {"philtrum_region_max_fraction"}, 0.32);
    if (key == "left_brow_ridge" || key == "right_brow_ridge") return getF(params, {"brow_region_max_fraction"}, 0.16);
    if (key == "highlight_candidate" || key == "treatable_skin") return getF(params, {"skin_region_max_fraction"}, 0.35);
    return 0.20;
}

// _remove_small_components()
cv::Mat removeSmallComponents(const cv::Mat& mask, int minArea) {
    cv::Mat bin;
    cv::compare(mask, 0, bin, cv::CMP_GT);  // 0/255
    cv::Mat binary = bin / 255;
    cv::Mat labels, stats, centroids;
    const int num = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    cv::Mat out = cv::Mat::zeros(mask.size(), CV_8U);
    std::vector<uchar> keep(num, 0);
    for (int i = 1; i < num; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minArea) keep[i] = 255;
    }
    for (int y = 0; y < mask.rows; ++y) {
        const int* pl = labels.ptr<int>(y);
        uchar* po = out.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) po[x] = keep[pl[x]];
    }
    return out;
}

std::vector<float> gather(const cv::Mat1f& values, const cv::Mat1b& mask) {
    std::vector<float> out;
    for (int y = 0; y < values.rows; ++y) {
        const float* pv = values[y];
        const uchar* pm = mask[y];
        for (int x = 0; x < values.cols; ++x) {
            if (pm[x]) out.push_back(pv[x]);
        }
    }
    return out;
}

struct AdaptiveResult {
    cv::Mat1b core;
    cv::Mat1b grown;
    cv::Mat1f score;
};

// _adaptive_region_mask()
AdaptiveResult adaptiveRegionMask(const std::string& key, const cv::Mat& regionMask, const cv::Mat1b& base,
                                  const cv::Mat1f& rgbMax, const cv::Mat1f& rgbRange, const cv::Mat1f& gray,
                                  const cv::Mat1f& v, const cv::Mat1f& s, const cv::Mat1f& l,
                                  const cv::Mat1f& chromaDelta, const cv::Mat1f& localDiff,
                                  const cv::Mat1f& largeDiff, const Section& params) {
    const cv::Size size = base.size();
    AdaptiveResult res;
    res.core = cv::Mat1b::zeros(size);
    res.grown = cv::Mat1b::zeros(size);
    res.score = cv::Mat1f::zeros(size);

    cv::Mat1b region(size);
    int regionCount = 0;
    for (int y = 0; y < size.height; ++y) {
        const uchar* pm = regionMask.ptr<uchar>(y);
        const uchar* pb = base[y];
        uchar* pr = region[y];
        for (int x = 0; x < size.width; ++x) {
            pr[x] = (pm[x] > 0 && pb[x]) ? 255 : 0;
            regionCount += pr[x] ? 1 : 0;
        }
    }
    if (regionCount < 20) return res;

    const auto lv = gather(l, region);
    const auto vv = gather(v, region);
    const auto sv = gather(s, region);
    const auto gv = gather(gray, region);
    const auto cdv = gather(chromaDelta, region);
    const auto ldv = gather(localDiff, region);

    const double corePct = getF(params, {"adaptive_core_percentile"}, 90);
    const double l50 = percentileOf(lv, 50, 185);
    const double l75 = percentileOf(lv, 75, l50 + 7);
    const double l90 = percentileOf(lv, corePct, l50 + 14);
    const double l95 = percentileOf(lv, 95, l90 + 5);
    const double v50 = percentileOf(vv, 50, 205);
    const double v75 = percentileOf(vv, 75, v50 + 7);
    const double v90 = percentileOf(vv, 90, v50 + 14);
    const double s50 = percentileOf(sv, 50, 95);
    const double s75 = percentileOf(sv, 75, s50 + 10);
    const double g50 = percentileOf(gv, 50, 175);
    const double g75 = percentileOf(gv, 75, g50 + 7);
    const double g90 = percentileOf(gv, 90, g50 + 14);
    const double cd75 = percentileOf(cdv, 75, 24);
    const double ld75 = percentileOf(ldv, 75, 2);
    const double ld90 = percentileOf(ldv, 90, 5);

    const double absLThr = getF(params, {"lab_l_threshold", "lab_l_thr"}, 188);
    const double absVThr = getF(params, {"hsv_v_threshold", "hsv_v_thr"}, 190);
    const double rgbThr = getF(params, {"rgb_brightness_threshold", "rgb_thr"}, 205);
    const double hsvSUpper = getF(params, {"hsv_s_upper"}, 150);
    const double oilSUpper = getF(params, {"oil_shine_s_upper"}, std::max(150.0, hsvSUpper));
    const double delta = getF(params, {"adaptive_region_delta"}, 4.8);
    const double coreDelta = getF(params, {"adaptive_core_delta"}, 8.0);
    const double relSatDrop = getF(params, {"relative_saturation_drop"}, -8.0);
    const double haloPct = getF(params, {"adaptive_halo_percentile"}, 78);

    double lHaloThr = std::max(percentileOf(lv, haloPct, l75), l50 + delta);
    double lCoreThr = std::max(percentileOf(lv, corePct, l90), l50 + coreDelta);
    double vHaloThr = std::max(percentileOf(vv, haloPct, v75), v50 + delta);
    double gHaloThr = std::max(percentileOf(gv, haloPct, g75), g50 + delta);

    if (key == "nose_bridge" || key == "nose_tip") {
        lHaloThr -= 2.0;
        vHaloThr -= 2.0;
        gHaloThr -= 2.0;
        lCoreThr -= 2.0;
    } else if (key == "left_brow_ridge" || key == "right_brow_ridge") {
        lHaloThr -= 2.0;
        vHaloThr -= 2.0;
        gHaloThr -= 2.0;
        lCoreThr -= 2.5;
    } else if (key == "left_cheek" || key == "right_cheek") {
        lHaloThr += 1.5;
        lCoreThr += 1.0;
    } else if (key == "chin") {
        lHaloThr += 1.0;
    }

    const double chromaUpper = getF(params, {"local_skin_chroma_delta_upper"}, 42);
    const double localThrParam = getF(params, {"local_brightness_threshold"}, 5);
    const double satPixThr = getF(params, {"saturation_pixel_threshold"}, 245);

    const float lNormDen = static_cast<float>(std::max(l95 - l50, 8.0));
    const float vNormDen = static_cast<float>(std::max(v90 - v50, 8.0));
    const float gNormDen = static_cast<float>(std::max(g90 - g50, 8.0));
    const float localNormDen = static_cast<float>(std::max(std::max(ld90, ld75 + 1.0), 3.5));
    const float whiteSThr = static_cast<float>(std::max(hsvSUpper, s50 + 30));
    const float chromaScoreThr = static_cast<float>(std::max(chromaUpper, cd75 + 14));
    const float warmSThr = static_cast<float>(std::max(oilSUpper, s75 + 42.0));
    const float ntcThr = static_cast<float>(std::max(chromaUpper, cd75 + 18.0));
    const float relLowSatThr = static_cast<float>(s50 - relSatDrop);
    const float localCoreThr = static_cast<float>(std::max(localThrParam, ld75 + 0.5));

    cv::Mat1b core = cv::Mat1b::zeros(size);
    cv::Mat1b halo = cv::Mat1b::zeros(size);
    cv::Mat1f& score = res.score;
    auto clip02 = [](float x) { return std::max(0.f, std::min(2.f, x)); };

    for (int y = 0; y < size.height; ++y) {
        const float* pl = l[y];
        const float* pv = v[y];
        const float* pg = gray[y];
        const float* ps = s[y];
        const float* prm = rgbMax[y];
        const float* prr = rgbRange[y];
        const float* pcd = chromaDelta[y];
        const float* pld = localDiff[y];
        const float* pgd = largeDiff[y];
        const uchar* pr = region[y];
        uchar* pc = core[y];
        uchar* ph = halo[y];
        float* psc = score[y];
        for (int x = 0; x < size.width; ++x) {
            const float lf = pl[x], vf = pv[x], gf = pg[x], sf = ps[x];
            const float rmx = prm[x], rrg = prr[x], cdf = pcd[x], ldf = pld[x], gdf = pgd[x];

            const float lNorm = (lf - static_cast<float>(l50)) / lNormDen;
            const float vNorm = (vf - static_cast<float>(v50)) / vNormDen;
            const float gNorm = (gf - static_cast<float>(g50)) / gNormDen;
            const float localNorm = std::max(ldf, gdf * 0.65f) / localNormDen;
            const float whiteScore = (rrg <= 62.f ? 0.55f : 0.f) + (sf <= whiteSThr ? 0.25f : 0.f);
            const float absScore = ((lf >= absLThr ? 1.f : 0.f) + (vf >= absVThr ? 1.f : 0.f) +
                                    (rmx >= rgbThr ? 1.f : 0.f)) / 3.f;
            const float chromaScore = cdf <= chromaScoreThr ? 1.f : 0.f;
            const float sc = 0.40f * clip02(lNorm) + 0.22f * clip02(vNorm) + 0.13f * clip02(gNorm) +
                             0.16f * clip02(localNorm) + 0.12f * whiteScore + 0.18f * absScore +
                             0.08f * chromaScore;
            psc[x] = sc;

            if (!pr[x]) continue;
            const bool warmAllowed = sf <= warmSThr;
            const bool notTooColored = cdf <= ntcThr;
            const bool relativeLowSat = sf <= relLowSatThr;

            const bool absoluteCore = ((lf >= absLThr && vf >= absVThr - 4 && warmAllowed) ||
                                       (rmx >= rgbThr && lf >= absLThr - 8 && warmAllowed) ||
                                       (rmx >= satPixThr - 3 && rrg <= 92));
            const bool percentileCore = lf >= lCoreThr && vf >= vHaloThr - 2 && warmAllowed;
            const bool localCore = ldf >= localCoreThr && lf >= lHaloThr - 2 && warmAllowed;
            if (notTooColored && (absoluteCore || percentileCore || localCore || (sc >= 0.74f && relativeLowSat))) {
                pc[x] = 255;
            }
            const bool haloHit =
                lf >= lHaloThr || vf >= vHaloThr || gf >= gHaloThr ||
                (sc >= 0.46f && lf >= l50 + delta * 0.55) ||
                (gdf >= std::max(2.0, localThrParam * 0.50) && lf >= l50 + delta * 0.5);
            if (notTooColored && warmAllowed && haloHit) ph[x] = 255;
        }
    }

    // 平滑油光 fallback 种子
    if (cv::countNonZero(core) == 0) {
        std::vector<float> rmRegion = gather(rgbMax, region);
        const double rm95 = percentileOf(rmRegion, 95, 0);
        if (l95 >= absLThr - 5 || v90 >= absVThr - 4 || rm95 >= rgbThr - 4) {
            const double seedPct = getF(params, {"fallback_seed_percentile"}, 93);
            std::vector<float> scRegion = gather(score, region);
            const double seedThr = percentileOf(scRegion, seedPct, 0.85);
            for (int y = 0; y < size.height; ++y) {
                const uchar* pr = region[y];
                const float* psc = score[y];
                const float* pl = l[y];
                const float* ps = s[y];
                const float* pcd = chromaDelta[y];
                uchar* pc = core[y];
                for (int x = 0; x < size.width; ++x) {
                    if (pr[x] && pcd[x] <= ntcThr && psc[x] >= seedThr && pl[x] >= lHaloThr - 1 &&
                        ps[x] <= warmSThr) {
                        pc[x] = 255;
                    }
                }
            }
        }
    }

    cv::Mat1b grown = growFromCore(core, halo, static_cast<int>(getF(params, {"adaptive_grow_iterations"}, 12)));
    const double maxFraction = std::max(0.03, std::min(0.95, regionFraction(key, params)));
    const int maxPixels = std::max(6, static_cast<int>(regionCount * maxFraction));
    cv::Mat1b grownOrCore(size);
    for (int y = 0; y < size.height; ++y) {
        const uchar* pg = grown[y];
        const uchar* pc = core[y];
        uchar* po = grownOrCore[y];
        for (int x = 0; x < size.width; ++x) po[x] = (pg[x] || pc[x]) ? 255 : 0;
    }
    res.grown = capByScore(grownOrCore, score, region, maxPixels);
    res.core = capByScore(core, score, region, std::max(3, static_cast<int>(maxPixels * 0.55)));
    return res;
}

}  // namespace

HighlightOutput detectHighlight(const cv::Mat& imageBgr, const RegionMasks& regions, const Section& params) {
    HighlightOutput out;
    const cv::Size size = imageBgr.size();
    const int h = size.height, w = size.width;

    const cv::Mat zeros = cv::Mat::zeros(size, CV_8U);
    const cv::Mat& skinM = regions.has("skin") ? regions.at("skin") : zeros;
    const cv::Mat& protectM = regions.has("protect") ? regions.at("protect") : zeros;
    const cv::Mat& candidateM = regions.has("highlight_candidate") ? regions.at("highlight_candidate") : zeros;

    cv::Mat1b base(size), protect(size);
    int baseCount = 0;
    for (int y = 0; y < h; ++y) {
        const uchar* pk = skinM.ptr<uchar>(y);
        const uchar* pp = protectM.ptr<uchar>(y);
        const uchar* pc = candidateM.ptr<uchar>(y);
        uchar* pb = base[y];
        uchar* pq = protect[y];
        for (int x = 0; x < w; ++x) {
            pq[x] = pp[x] > 0 ? 255 : 0;
            pb[x] = (pk[x] > 0 && pc[x] > 0 && pp[x] == 0) ? 255 : 0;
            baseCount += pb[x] ? 1 : 0;
        }
    }
    if (baseCount == 0) {
        out.hard_mask = zeros.clone();
        out.soft_mask = zeros.clone();
        out.warnings.push_back("高光候选区域为空，未执行高光检测");
        return out;
    }

    cv::Mat hsv, lab, gray8;
    cv::cvtColor(imageBgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(imageBgr, lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(imageBgr, gray8, cv::COLOR_BGR2GRAY);

    cv::Mat1f rgbMax(size), rgbRange(size), gray(size), v(size), s(size), l(size), a(size), b(size);
    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* pi = imageBgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b* ph = hsv.ptr<cv::Vec3b>(y);
        const cv::Vec3b* pl = lab.ptr<cv::Vec3b>(y);
        const uchar* pg = gray8.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const int bC = pi[x][0], gC = pi[x][1], rC = pi[x][2];
            const int mx = std::max({bC, gC, rC});
            const int mn = std::min({bC, gC, rC});
            rgbMax(y, x) = static_cast<float>(mx);
            rgbRange(y, x) = static_cast<float>(mx - mn);
            gray(y, x) = static_cast<float>(pg[x]);
            v(y, x) = static_cast<float>(ph[x][2]);
            s(y, x) = static_cast<float>(ph[x][1]);
            l(y, x) = static_cast<float>(pl[x][0]);
            a(y, x) = static_cast<float>(pl[x][1]);
            b(y, x) = static_cast<float>(pl[x][2]);
        }
    }

    const double rgbThr = getF(params, {"rgb_brightness_threshold", "rgb_thr"}, 205);
    const double hsvVThr = getF(params, {"hsv_v_threshold", "hsv_v_thr"}, 190);
    const double hsvSUpper = getF(params, {"hsv_s_upper"}, 150);
    const double oilSUpper = getF(params, {"oil_shine_s_upper"}, std::max(165.0, hsvSUpper));
    const double labLThr = getF(params, {"lab_l_threshold", "lab_l_thr"}, 188);
    const double satThr = getF(params, {"saturation_pixel_threshold", "saturated_pixel_threshold"}, 245);
    const double localThr = getF(params, {"local_brightness_threshold", "local_brightness_delta"}, 5);
    const double localRatioThr = getF(params, {"local_contrast_threshold"}, 0.025);

    const cv::Mat1f localL = maskedGaussian(l, base, getF(params, {"local_sigma"}, 6.0));
    const cv::Mat1f largeL = maskedGaussian(l, base, getF(params, {"large_local_sigma"}, 28.0));
    cv::Mat1f localDiff(size), largeDiff(size);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            localDiff(y, x) = l(y, x) - localL(y, x);
            largeDiff(y, x) = l(y, x) - largeL(y, x);
        }
    }

    // 肤色邻域色差
    int win = static_cast<int>(getF(params, {"local_window_radius"}, 31));
    win = std::max(7, win % 2 == 1 ? win : win + 1);
    cv::Mat1f baseF(size), aProd(size), bProd(size);
    for (int y = 0; y < h; ++y) {
        const uchar* pb = base[y];
        for (int x = 0; x < w; ++x) {
            const float bf = pb[x] ? 1.f : 0.f;
            baseF(y, x) = bf;
            aProd(y, x) = a(y, x) * bf;
            bProd(y, x) = b(y, x) * bf;
        }
    }
    cv::Mat1f wSum, aLocal, bLocal;
    cv::GaussianBlur(baseF, wSum, cv::Size(win, win), 0);
    cv::GaussianBlur(aProd, aLocal, cv::Size(win, win), 0);
    cv::GaussianBlur(bProd, bLocal, cv::Size(win, win), 0);
    cv::Mat1f chromaDelta(size);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float ws = wSum(y, x) + 1e-3f;
            const float da = a(y, x) - aLocal(y, x) / ws;
            const float db = b(y, x) - bLocal(y, x) / ws;
            chromaDelta(y, x) = std::sqrt(da * da + db * db);
        }
    }
    const double chromaUpper = getF(params, {"local_skin_chroma_delta_upper"}, 42);

    // 传统强证据
    cv::Mat1b classic = cv::Mat1b::zeros(size);
    for (int y = 0; y < h; ++y) {
        const uchar* pb = base[y];
        uchar* pcl = classic[y];
        for (int x = 0; x < w; ++x) {
            if (!pb[x]) continue;
            if (!(chromaDelta(y, x) <= chromaUpper)) continue;
            const float rmx = rgbMax(y, x), rrg = rgbRange(y, x);
            const float vf = v(y, x), sf = s(y, x), lf = l(y, x);
            const float ldf = localDiff(y, x), gdf = largeDiff(y, x);
            const float ratio = ldf / std::max(localL(y, x), 1.f);
            const bool ruleRgb = rmx >= rgbThr || (rmx >= satThr && rrg <= 96);
            const bool ruleHsvWhite = vf >= hsvVThr && sf <= hsvSUpper;
            const bool ruleHsvOil = vf >= hsvVThr - 8 && sf <= oilSUpper;
            const bool ruleLab = lf >= labLThr;
            const bool ruleLocal = ldf >= localThr || ratio >= localRatioThr || gdf >= std::max(2.0, localThr * 0.45);
            const int scoreCount = ruleRgb + ruleHsvWhite + ruleHsvOil + ruleLab + ruleLocal;
            if ((scoreCount >= 2 && (ruleLab || ruleHsvOil || ruleLocal)) ||
                (scoreCount >= 1 && ruleLocal && (ruleRgb || ruleLab || ruleHsvOil))) {
                pcl[x] = 255;
            }
        }
    }

    cv::Mat1b adaptiveCore = cv::Mat1b::zeros(size);
    cv::Mat1b adaptiveGrown = cv::Mat1b::zeros(size);
    cv::Mat1f globalScore = cv::Mat1f::zeros(size);

    std::vector<std::string> regionKeys;
    if (regions.has("treatable_skin")) {
        regionKeys = {"highlight_candidate"};
    } else {
        regionKeys = {"forehead", "nose_bridge", "nose_tip",   "philtrum",    "left_brow_ridge",
                      "right_brow_ridge", "left_cheek", "right_cheek", "chin"};
    }
    for (const auto& key : regionKeys) {
        if (!regions.has(key)) continue;
        const AdaptiveResult res = adaptiveRegionMask(key, regions.at(key), base, rgbMax, rgbRange, gray, v, s, l,
                                                      chromaDelta, localDiff, largeDiff, params);
        for (int y = 0; y < h; ++y) {
            const uchar* pc = res.core[y];
            const uchar* pg = res.grown[y];
            const float* ps = res.score[y];
            uchar* pac = adaptiveCore[y];
            uchar* pag = adaptiveGrown[y];
            float* pgs = globalScore[y];
            for (int x = 0; x < w; ++x) {
                if (pc[x]) pac[x] = 255;
                if (pg[x]) pag[x] = 255;
                pgs[x] = std::max(pgs[x], ps[x]);
            }
        }
    }

    cv::Mat hard = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < h; ++y) {
        const uchar* pb = base[y];
        const uchar* pcl = classic[y];
        const uchar* pac = adaptiveCore[y];
        const uchar* pag = adaptiveGrown[y];
        uchar* ph = hard.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (pb[x] && (pcl[x] || pac[x] || pag[x])) ph[x] = 255;
        }
    }

    const int closeR = static_cast<int>(getF(params, {"morph_close_radius"}, 3));
    const int dilateR = static_cast<int>(getF(params, {"mask_dilate_radius", "morph_dilate_radius"}, 1));
    const int erodeR = static_cast<int>(getF(params, {"mask_erode_radius", "morph_erode_radius"}, 0));
    const int blurR = static_cast<int>(getF(params, {"mask_blur_radius"}, 9));

    if (closeR > 0) {
        const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * closeR + 1, 2 * closeR + 1));
        cv::morphologyEx(hard, hard, cv::MORPH_CLOSE, k);
    }
    if (erodeR > 0) hard = morphMask(hard, "erode", erodeR);
    if (dilateR > 0) hard = morphMask(hard, "dilate", dilateR);

    for (int y = 0; y < h; ++y) {
        const uchar* pb = base[y];
        const uchar* pq = protect[y];
        uchar* ph = hard.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (!pb[x] || pq[x]) ph[x] = 0;
        }
    }
    hard = removeSmallComponents(hard, static_cast<int>(getF(params, {"highlight_min_area", "min_area"}, 8)));

    // 面积上限截断
    const cv::Mat& faceAreaMask = regions.has("face_mask") ? regions.at("face_mask")
                                 : (regions.has("skin") ? regions.at("skin") : hard);
    const int faceArea = std::max(1, cv::countNonZero(faceAreaMask));
    const int maxArea = static_cast<int>(faceArea * getF(params, {"highlight_max_area_ratio", "max_area_ratio"}, 0.125));
    if (maxArea > 0 && cv::countNonZero(hard) > maxArea) {
        cv::Mat1f strength(size);
        auto clip03 = [](float x) { return std::max(0.f, std::min(3.f, x)); };
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                strength(y, x) = globalScore(y, x) +
                                 0.18f * clip03((l(y, x) - static_cast<float>(labLThr)) / 20.f) +
                                 0.12f * clip03((v(y, x) - static_cast<float>(hsvVThr)) / 25.f) +
                                 0.10f * clip03(localDiff(y, x) / static_cast<float>(std::max(localThr, 1.0)));
            }
        }
        std::vector<float> vals;
        for (int y = 0; y < h; ++y) {
            const uchar* ph = hard.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                if (ph[x] > 0) vals.push_back(strength(y, x));
            }
        }
        if (static_cast<int>(vals.size()) > maxArea) {
            const float kth = kthLargest(vals, maxArea);
            for (int y = 0; y < h; ++y) {
                const uchar* pb = base[y];
                const uchar* pq = protect[y];
                uchar* ph = hard.ptr<uchar>(y);
                for (int x = 0; x < w; ++x) {
                    ph[x] = (ph[x] > 0 && strength(y, x) >= kth && pb[x] && !pq[x]) ? 255 : 0;
                }
            }
        }
    }

    if (cv::countNonZero(hard) == 0) {
        out.warnings.push_back("未检测到满足条件的高光区域，可降低 HSV V、Lab L、局部亮度异常阈值或提高油光饱和度上限");
    }

    cv::Mat soft = blurR > 0 ? blurMask(hard, blurR) : hard.clone();
    const float softGain = static_cast<float>(getF(params, {"soft_mask_gain"}, 1.65));
    for (int y = 0; y < h; ++y) {
        const uchar* pb = base[y];
        const uchar* pq = protect[y];
        uchar* ps = soft.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const float scaled = std::min(255.f, std::max(0.f, static_cast<float>(ps[x]) * softGain));
            ps[x] = truncU8(scaled);
            if (!pb[x] || pq[x]) ps[x] = 0;
        }
    }

    out.hard_mask = hard;
    out.soft_mask = soft;
    return out;
}

}  // namespace hr
