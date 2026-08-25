#include "high_removal/pipeline.hpp"

#include <opencv2/imgproc.hpp>

namespace hr {

namespace {

struct FaceCandidate {
    Landmarks landmarks;  // 原图坐标系
    Bbox bbox;
    float confidence = 0.f;
};

// scale_landmarks()
Landmarks scaleLandmarks(const Landmarks& lm, double scale) {
    Landmarks out = lm;
    for (auto& p : out) {
        p[0] = static_cast<float>(p[0] * scale);
        p[1] = static_cast<float>(p[1] * scale);
        p[2] = static_cast<float>(p[2] * scale);
    }
    return out;
}

// _merge_removal_params()：highlight_removal ∪ highlight_detection（detection 覆盖），
// 加上 pipeline 的 ROI 设置。
Section mergedRemovalParams(const Config& config) {
    Section merged = config.highlight_removal;
    for (const auto& [k, v] : config.highlight_detection) merged[k] = v;
    merged["enable_roi_removal"] = getB(config.pipeline, {"enable_roi_removal"}, true);
    merged["roi_padding_ratio"] = getF(config.pipeline, {"roi_padding_ratio"}, 0.12);
    return merged;
}

}  // namespace

PipelineOutput processImage(const cv::Mat& imageBgr, const Config& config, const FaceLandmarkerOrt& landmarker) {
    PipelineOutput out;
    const ProcessScales scales = resolveProcessScales(config.pipeline);
    const bool refine = getB(config.pipeline, {"refine_upsampled_masks"}, true);
    if (getB(config.face_detection, {"enable_alignment"}, false)) {
        out.warnings.push_back("警告：C++ 版未实现 enable_alignment 分支（默认配置为关闭），按未对齐路径处理");
    }

    // 1) 人脸关键点（face_scale 分辨率），映射回原图。
    const cv::Mat detImage = resizeForProcess(imageBgr, scales.face);
    const auto faces = landmarker.detect(detImage);
    if (faces.empty()) {
        out.status = "未检测到可靠人脸，请更换图片或调整检测参数";
        out.result_bgr = imageBgr.clone();
        return out;
    }
    const double minConf = getF(config.face_detection, {"face_detection_confidence"}, 0.55);
    const double invScale = scales.face >= 0.999 ? 1.0 : 1.0 / clampProcessScale(scales.face);
    std::vector<FaceCandidate> candidates;
    for (const auto& f : faces) {
        FaceCandidate c;
        c.landmarks = invScale == 1.0 ? f.landmarks : scaleLandmarks(f.landmarks, invScale);
        c.bbox = bboxFromLandmarks(c.landmarks, imageBgr.size());
        c.confidence = std::max(static_cast<float>(minConf), 0.90f);  // _detect_mediapipe_tasks()
        candidates.push_back(std::move(c));
    }
    // select_face()：默认策略「最大人脸」= max(面积, 置信度)。
    const std::string strategy = getS(config.face_detection, {"multi_face_strategy"}, "最大人脸");
    size_t selectedIdx = 0;
    if (strategy == "手动编号") {
        const int manual = static_cast<int>(getF(config.face_detection, {"manual_face_index"}, 0));
        selectedIdx = static_cast<size_t>(std::max(0, std::min<int>(manual, static_cast<int>(candidates.size()) - 1)));
    } else if (strategy == "置信度最高") {
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].confidence > candidates[selectedIdx].confidence) selectedIdx = i;
        }
    } else {
        auto area = [](const Bbox& b) { return static_cast<long long>(std::max(0, b.w)) * std::max(0, b.h); };
        for (size_t i = 1; i < candidates.size(); ++i) {
            const auto ai = area(candidates[i].bbox), as = area(candidates[selectedIdx].bbox);
            if (ai > as || (ai == as && candidates[i].confidence > candidates[selectedIdx].confidence)) selectedIdx = i;
        }
    }
    const Landmarks& lm = candidates[selectedIdx].landmarks;
    if (candidates.size() > 1) out.warnings.push_back("检测到多张人脸，已默认处理最大人脸");

    // 2) 高光检测（_detect_highlight_masks）。
    Section hdParams = config.highlight_detection;
    const Section& regionParams = config.regions;
    cv::Mat hardFull, softFull;
    RegionMasks regions;
    const bool useFullHighlight = scales.highlight >= 0.999 || refine;
    if (useFullHighlight) {
        // 原分辨率检测（refine 或 process_scale=1）。refine 的 ROI 重检测分支未移植：
        // default.yaml 中 refine_upsampled_masks=false，不会走到这里，除非用户改配置。
        regions = buildFaceRegions(imageBgr, lm, regionParams);
        const HighlightOutput highlight = detectHighlight(imageBgr, regions, hdParams);
        for (const auto& warn : highlight.warnings) out.warnings.push_back(warn);
        hardFull = highlight.hard_mask;
        softFull = highlight.soft_mask;
    } else {
        const cv::Mat small = resizeForProcess(imageBgr, scales.highlight);
        const Landmarks lmSmall = scaleLandmarks(lm, scales.highlight);
        const RegionMasks regionsSmall = buildFaceRegions(small, lmSmall, regionParams);
        const Section hdScaled = scaledHighlightParams(hdParams, scales.highlight);
        const HighlightOutput highlight = detectHighlight(small, regionsSmall, hdScaled);
        for (const auto& warn : highlight.warnings) out.warnings.push_back(warn);
        hardFull = upsampleMaskHard(highlight.hard_mask, imageBgr.size());
        softFull = upsampleMaskSoft(highlight.soft_mask, imageBgr.size());
        regions = buildFaceRegions(imageBgr, lm, regionParams);
    }
    for (const auto& warn : regions.warnings) out.warnings.push_back(warn);

    // 3) 去高光修复。
    const RemovalOutput removal = removeHighlight(imageBgr, hardFull, softFull, regions, mergedRemovalParams(config));
    for (const auto& warn : removal.warnings) out.warnings.push_back(warn);

    out.success = true;
    out.status = out.warnings.empty() ? "处理成功" : "处理完成，存在需要人工复核的警告";
    out.result_bgr = removal.result_bgr;
    out.highlight_mask = hardFull;
    out.soft_mask = softFull;
    return out;
}

}  // namespace hr
