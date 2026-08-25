// ONNX Runtime 实现的 MediaPipe FaceLandmarker（IMAGE 模式）等价物。
// 与 tools/onnx_landmarker_ref.py 一一对应；前后处理按 MediaPipe CPU 源码复刻：
//   ImageToTensor(warpPerspective) → BlazeFace 解码(896 anchors) → 加权 NMS(IoU 0.3)
//   → 双眼旋转对齐 + 1.5x 方形扩框 → 256x256 裁剪 → 478 点反投影。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "high_removal/common.hpp"

namespace hr {

struct DetectedFace {
    Landmarks landmarks;  // 像素坐标：x*w, y*h, z*max(w,h)（与 face_detect.py 一致）
    float presence = 0.f;
};

class FaceLandmarkerOrt {
public:
    FaceLandmarkerOrt(const std::string& modelDir, int numFaces,
                      float minDetectionConfidence, float minPresenceConfidence);
    // 从内存字节加载两个 ONNX 模型（供单 ONNX 自定义算子使用：模型以节点属性内嵌，无需磁盘文件）。
    FaceLandmarkerOrt(const void* detectorData, size_t detectorSize,
                      const void* landmarksData, size_t landmarksSize,
                      int numFaces, float minDetectionConfidence, float minPresenceConfidence);
    ~FaceLandmarkerOrt();

    std::vector<DetectedFace> detect(const cv::Mat& imageBgr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hr
