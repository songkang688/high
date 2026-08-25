// libhigh_removal_ops：ONNX Runtime 自定义算子库（微软官方
// "Wrapping an external inference runtime in a custom operator" 模式，
// 参见 https://onnxruntime.ai/docs/reference/operators/add-custom-op.html）。
//
// models/high_removal.onnx 的图只有一个 ai.high:HighlightRemoval 节点：
//   输入  image     uint8 [H, W, 3]（BGR，与 cv2.imread 一致）
//   输出  result    uint8 [H, W, 3]（去高光结果，BGR）
//   输出  hard_mask uint8 [H, W]  （高光硬掩码，0/255）
// 两个人脸 ONNX 模型与 configs/default.yaml 文本以节点属性内嵌（uint8 张量），
// 内核反序列化后运行已对拍验证过的 hr::processImage C++ 流水线。
//
// 本库按官方推荐不直接链接 libonnxruntime：OrtApi 由宿主在 RegisterCustomOps 时传入
//（ORT_API_MANUAL_INIT），因此同一个 .so 可以被 Python onnxruntime 或任意 C++ ORT 应用加载。
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "high_removal/pipeline.hpp"

namespace {

constexpr const char* kDomain = "ai.high";
constexpr const char* kOpName = "HighlightRemoval";

// 节点属性名（与 tools/export_high_removal_onnx.py 保持一致）。
constexpr const char* kAttrDetectorModel = "face_detector_model";
constexpr const char* kAttrLandmarksModel = "face_landmarks_model";
constexpr const char* kAttrConfigYaml = "config_yaml";

std::string readBytesAttribute(const Ort::ConstKernelInfo& info, const char* name) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::Value value = info.GetTensorAttribute(name, alloc);
    const size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* data = value.GetTensorData<uint8_t>();
    return std::string(reinterpret_cast<const char*>(data), count);
}

OrtStatusPtr statusFromException(const std::exception& exc, OrtErrorCode code) {
    return Ort::GetApi().CreateStatus(code, exc.what());
}

struct HighlightRemovalKernel {
    hr::Config config;
    int numFaces = 4;
    float minDetConf = 0.55f;
    float minPresence = 0.50f;

    // 人脸 ONNX 会话延迟到第一次 Compute 再创建：避免在宿主会话初始化阶段嵌套建会话。
    std::string detectorBytes;
    std::string landmarksBytes;
    std::once_flag landmarkerOnce;
    std::unique_ptr<hr::FaceLandmarkerOrt> landmarker;

    explicit HighlightRemovalKernel(const OrtKernelInfo* info) {
        const Ort::ConstKernelInfo kinfo{info};
        detectorBytes = readBytesAttribute(kinfo, kAttrDetectorModel);
        landmarksBytes = readBytesAttribute(kinfo, kAttrLandmarksModel);
        config = hr::loadConfigFromString(readBytesAttribute(kinfo, kAttrConfigYaml));
        numFaces = static_cast<int>(hr::getF(config.face_detection, {"max_faces"}, 4));
        minDetConf = static_cast<float>(hr::getF(config.face_detection, {"face_detection_confidence"}, 0.55));
        minPresence = static_cast<float>(hr::getF(config.face_detection, {"landmark_detection_confidence"}, 0.50));
    }

    void ensureLandmarker() {
        std::call_once(landmarkerOnce, [this] {
            landmarker = std::make_unique<hr::FaceLandmarkerOrt>(
                detectorBytes.data(), detectorBytes.size(), landmarksBytes.data(), landmarksBytes.size(),
                numFaces, minDetConf, minPresence);
            std::string().swap(detectorBytes);  // 会话已持有解析后的模型，释放原始字节
            std::string().swap(landmarksBytes);
        });
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
        try {
            ensureLandmarker();

            const Ort::KernelContext ctx{context};
            Ort::ConstValue input = ctx.GetInput(0);
            const std::vector<int64_t> shape = input.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[2] != 3 || shape[0] <= 0 || shape[1] <= 0) {
                return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                                  "HighlightRemoval: image 输入必须是 [H, W, 3] 的 uint8 BGR 张量");
            }
            const auto rows = static_cast<int>(shape[0]);
            const auto cols = static_cast<int>(shape[1]);
            const auto* pixels = input.GetTensorData<uint8_t>();
            const cv::Mat image(rows, cols, CV_8UC3, const_cast<uint8_t*>(pixels));

            const hr::PipelineOutput out = hr::processImage(image, config, *landmarker);
            for (const auto& warn : out.warnings) std::cerr << "[high_removal_ops] " << warn << "\n";
            if (!out.success) std::cerr << "[high_removal_ops] " << out.status << "\n";

            // 与 Python pipeline 一致：失败（如未检测到人脸）时原样返回输入图、掩码全零。
            Ort::UnownedValue result = ctx.GetOutput(0, shape.data(), shape.size());
            cv::Mat resultWrap(rows, cols, CV_8UC3, result.GetTensorMutableData<uint8_t>());
            (out.result_bgr.empty() ? image : out.result_bgr).copyTo(resultWrap);

            const std::array<int64_t, 2> maskShape{shape[0], shape[1]};
            Ort::UnownedValue mask = ctx.GetOutput(1, maskShape.data(), maskShape.size());
            cv::Mat maskWrap(rows, cols, CV_8UC1, mask.GetTensorMutableData<uint8_t>());
            if (out.highlight_mask.empty()) {
                maskWrap.setTo(0);
            } else {
                out.highlight_mask.copyTo(maskWrap);
            }
            return nullptr;
        } catch (const Ort::Exception& exc) {
            return statusFromException(exc, ORT_RUNTIME_EXCEPTION);
        } catch (const std::exception& exc) {
            return statusFromException(exc, ORT_RUNTIME_EXCEPTION);
        }
    }
};

struct HighlightRemovalOp : Ort::CustomOpBase<HighlightRemovalOp, HighlightRemovalKernel, /*WithStatus=*/true> {
    OrtStatusPtr CreateKernelV2(const OrtApi& /*api*/, const OrtKernelInfo* info, void** kernel) const noexcept {
        try {
            *kernel = new HighlightRemovalKernel(info);
            return nullptr;
        } catch (const std::exception& exc) {
            return statusFromException(exc, ORT_RUNTIME_EXCEPTION);
        }
    }
    const char* GetName() const { return kOpName; }
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
    size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
};

}  // namespace

// ORT 加载自定义算子库时调用的唯一入口（onnxruntime_c_api.h RegisterCustomOpsFn 约定）。
// Windows 侧导出由 src/high_removal_ops.def（EXPORTS RegisterCustomOps）声明，
// 与 Linux 的 version script 对应；ELF 侧仍需显式 default visibility。
#ifdef _WIN32
#define HR_ORT_EXPORT
#else
#define HR_ORT_EXPORT __attribute__((visibility("default")))
#endif
extern "C" HR_ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* apiBase) {
    const OrtApi* api = apiBase->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        // 宿主 ORT 版本过旧，连本库编译所用的 API 版本都不支持；无法构造 OrtStatus，只能中止。
        std::cerr << "[high_removal_ops] 宿主 ONNX Runtime 不支持 API 版本 " << ORT_API_VERSION
                  << "，请使用 >= 1.22 的 onnxruntime\n";
        std::abort();
    }
    Ort::InitApi(api);

    static HighlightRemovalOp op;  // 生命周期需覆盖所有会话
    OrtCustomOpDomain* domain = nullptr;
    if (OrtStatus* status = api->CreateCustomOpDomain(kDomain, &domain)) return status;
    if (OrtStatus* status = api->CustomOpDomain_Add(domain, &op)) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }
    if (OrtStatus* status = api->AddCustomOpDomain(options, domain)) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }
    // AddCustomOpDomain 不转移所有权，域对象必须存活到进程结束（官方示例同样用静态容器保活）。
    static std::vector<OrtCustomOpDomain*> keepAlive;
    static std::mutex keepAliveMutex;
    {
        const std::lock_guard<std::mutex> lock(keepAliveMutex);
        keepAlive.push_back(domain);
    }
    return nullptr;
}
