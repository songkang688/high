// facehi ONNX Runtime 自定义算子库：ai.facehi:HighlightRemoval。
//
// 官方做法（onnxruntime.ai/docs/reference/operators/add-custom-op.html，
// “Wrapping an external inference runtime in a custom operator”）：
// 一个 facehi.onnx 图中只有一个自定义节点，把整条去高光流水线
// （BlazeFace 检测 → 478 点关键点 → 分区 → 高光检测 → 修复 → 保护回写）
// 全部包进该节点。两个子模型（face_detector / face_landmarks_detector）、
// 合并后的配置 YAML、Haar 兜底级联 XML 全部序列化进节点属性，
// 用户只需 facehi.onnx + 本共享库即可运行，无需携带 configs/ 与子模型文件。
//
// 输入：
//   image      uint8  [H,W,3] 或 [1,H,W,3]，BGR（与 cv2.imread 一致）
//   landmarks  float32 [478,3]（可选，默认空 [0,3]；注入后跳过人脸检测，
//              用于黄金对照 / 复用外部关键点）
// 输出：
//   result          uint8 同 image 形状（去高光结果）
//   highlight_mask  uint8 [H,W]（或 [1,H,W]，随输入秩）最终硬 mask
//
// 未检测到人脸时与 Python pipeline 失败分支一致：result 返回原图、mask 全零。
#include <cstring>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "facehi/config.hpp"
#include "facehi/face_detect.hpp"
#include "facehi/pipeline.hpp"

namespace {

constexpr const char* kDomain = "ai.facehi";

std::string tensor_attr_as_bytes(const Ort::ConstKernelInfo& info, const char* name,
                                 OrtAllocator* alloc, bool required) {
  try {
    Ort::Value v = info.GetTensorAttribute(name, alloc);
    auto shape_info = v.GetTensorTypeAndShapeInfo();
    size_t n = shape_info.GetElementCount();
    const uint8_t* p = v.GetTensorData<uint8_t>();
    return std::string(reinterpret_cast<const char*>(p), n);
  } catch (const Ort::Exception&) {
    if (required) throw;
    return {};
  }
}

std::string string_attr_or(const Ort::ConstKernelInfo& info, const char* name,
                           const std::string& def) {
  try {
    return info.GetAttribute<std::string>(name);
  } catch (const Ort::Exception&) {
    return def;
  }
}

struct HighlightRemovalKernel {
  explicit HighlightRemovalKernel(const OrtKernelInfo* kernel_info) {
    Ort::ConstKernelInfo info{kernel_info};
    Ort::AllocatorWithDefaultOptions alloc;

    std::string detector = tensor_attr_as_bytes(info, "detector_onnx", alloc, /*required=*/true);
    std::string landmarks = tensor_attr_as_bytes(info, "landmarks_onnx", alloc, /*required=*/true);
    std::string haar_xml = tensor_attr_as_bytes(info, "haar_xml", alloc, /*required=*/false);
    std::string config_yaml = info.GetAttribute<std::string>("config_yaml");
    std::string mode = string_attr_or(info, "mode", "");

    config_ = facehi::load_embedded_config(config_yaml, mode);
    if (!haar_xml.empty()) facehi::set_haar_cascade_xml(haar_xml);

    double min_conf = config_.face_detection.getd("face_detection_confidence", 0.55);
    double lm_conf = config_.face_detection.getd("landmark_detection_confidence", 0.50);
    int max_faces = config_.face_detection.geti("max_faces", 4);
    landmarker_ = std::make_unique<facehi::OnnxFaceLandmarker>(
        detector.data(), detector.size(), landmarks.data(), landmarks.size(), min_conf, lm_conf,
        max_faces);
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* context) {
    try {
      Ort::KernelContext ctx(context);

      Ort::ConstValue image_in = ctx.GetInput(0);
      auto image_info = image_in.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> dims = image_info.GetShape();
      bool has_batch = dims.size() == 4;
      if (!(dims.size() == 3 || has_batch) || dims.back() != 3 ||
          (has_batch && dims[0] != 1)) {
        return Ort::GetApi().CreateStatus(
            ORT_INVALID_ARGUMENT,
            "image 输入形状必须是 [H,W,3] 或 [1,H,W,3]（uint8 BGR）");
      }
      int h = static_cast<int>(dims[has_batch ? 1 : 0]);
      int w = static_cast<int>(dims[has_batch ? 2 : 1]);
      cv::Mat bgr(h, w, CV_8UC3,
                  const_cast<uint8_t*>(image_in.GetTensorData<uint8_t>()));

      // 可选注入关键点（默认 initializer 为空 [0,3]）。
      cv::Mat injected;
      Ort::ConstValue lm_in = ctx.GetInput(1);
      auto lm_info = lm_in.GetTensorTypeAndShapeInfo();
      if (lm_info.GetElementCount() > 0) {
        std::vector<int64_t> ld = lm_info.GetShape();
        int rows = static_cast<int>(ld[ld.size() == 3 ? 1 : 0]);
        int cols = static_cast<int>(ld.back());
        if (cols != 3 || rows < 300)
          return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                            "landmarks 输入形状必须是 [478,3] float32");
        injected = cv::Mat(rows, 3, CV_32F,
                           const_cast<float*>(lm_in.GetTensorData<float>()))
                       .clone();
      }

      facehi::PipelineOutput out = facehi::process_image(
          bgr, config_, landmarker_.get(), injected.empty() ? nullptr : &injected);

      Ort::UnownedValue result_out = ctx.GetOutput(0, dims.data(), dims.size());
      uint8_t* rp = result_out.GetTensorMutableData<uint8_t>();
      cv::Mat result = out.result_bgr.isContinuous() ? out.result_bgr : out.result_bgr.clone();
      std::memcpy(rp, result.data, static_cast<size_t>(h) * w * 3);

      std::vector<int64_t> mask_dims;
      if (has_batch) mask_dims = {1, h, w};
      else mask_dims = {h, w};
      Ort::UnownedValue mask_out = ctx.GetOutput(1, mask_dims.data(), mask_dims.size());
      uint8_t* mp = mask_out.GetTensorMutableData<uint8_t>();
      cv::Mat mask = out.highlight_mask.isContinuous() ? out.highlight_mask
                                                       : out.highlight_mask.clone();
      std::memcpy(mp, mask.data, static_cast<size_t>(h) * w);
      return nullptr;
    } catch (const std::exception& e) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
    }
  }

 private:
  facehi::Config config_;
  std::unique_ptr<facehi::OnnxFaceLandmarker> landmarker_;
};

struct HighlightRemovalOp
    : Ort::CustomOpBase<HighlightRemovalOp, HighlightRemovalKernel, /*WithStatus=*/true> {
  OrtStatusPtr CreateKernelV2(const OrtApi& /*api*/, const OrtKernelInfo* info,
                              void** op_kernel) const {
    try {
      *op_kernel = new HighlightRemovalKernel(info);
      return nullptr;
    } catch (const std::exception& e) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
    }
  }

  const char* GetName() const { return "HighlightRemoval"; }

  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }
};

}  // namespace

// ORT 官方共享库约定：导出 RegisterCustomOps(OrtSessionOptions*, const OrtApiBase*)。
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  static const HighlightRemovalOp op;
  OrtStatus* status = nullptr;
  try {
    static Ort::CustomOpDomain domain = [] {
      Ort::CustomOpDomain d{kDomain};
      d.Add(&op);
      return d;
    }();
    status = Ort::GetApi().AddCustomOpDomain(options, domain);
  } catch (const std::exception& e) {
    status = Ort::GetApi().CreateStatus(ORT_FAIL, e.what());
  }
  return status;
}
