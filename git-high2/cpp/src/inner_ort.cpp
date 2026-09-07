#include "facehi/inner_ort.hpp"

#include <stdexcept>
#include <string>

#include <onnxruntime_c_api.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <libgen.h>
#endif

#include <filesystem>

namespace facehi {
namespace {

void throw_status(const OrtApi* api, OrtStatus* st, const char* what) {
  if (!st) return;
  std::string msg = what;
  msg += ": ";
  msg += api->GetErrorMessage(st);
  api->ReleaseStatus(st);
  throw std::runtime_error(msg);
}

std::filesystem::path module_dir() {
#ifdef _WIN32
  HMODULE self = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&module_dir), &self)) {
    throw std::runtime_error("GetModuleHandleExW 失败，无法定位自定义算子库目录");
  }
  wchar_t buf[MAX_PATH];
  DWORD n = GetModuleFileNameW(self, buf, MAX_PATH);
  if (n == 0 || n >= MAX_PATH)
    throw std::runtime_error("GetModuleFileNameW 失败");
  return std::filesystem::path(buf).parent_path();
#else
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&module_dir), &info) || !info.dli_fname)
    throw std::runtime_error("dladdr 失败，无法定位自定义算子库目录");
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

const OrtApi* load_inner_api() {
  static const OrtApi* api = []() -> const OrtApi* {
#ifdef _WIN32
    const auto path = module_dir() / L"facehi_ort122.dll";
    HMODULE h = LoadLibraryW(path.c_str());
    if (!h) {
      throw std::runtime_error(
          "找不到 facehi_ort122.dll（应与 facehi_custom_ops.dll 同目录）。"
          "这是 ORT 1.22.0 私有副本，供自定义算子内部跑人脸检测子模型。");
    }
    using GetBase = const OrtApiBase* (*)();
    auto get_base = reinterpret_cast<GetBase>(GetProcAddress(h, "OrtGetApiBase"));
    if (!get_base) throw std::runtime_error("facehi_ort122.dll 缺少 OrtGetApiBase");
    const OrtApi* p = get_base()->GetApi(ORT_API_VERSION);
    if (!p) throw std::runtime_error("facehi_ort122.dll GetApi(22) 失败");
    return p;
#else
    const auto path = module_dir() / "libfacehi_ort122.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!h) {
      throw std::runtime_error(
          std::string("找不到 libfacehi_ort122.so（应与 libfacehi_custom_ops.so 同目录）: ") +
          dlerror());
    }
    using GetBase = const OrtApiBase* (*)();
    auto get_base = reinterpret_cast<GetBase>(dlsym(h, "OrtGetApiBase"));
    if (!get_base) throw std::runtime_error("libfacehi_ort122.so 缺少 OrtGetApiBase");
    const OrtApi* p = get_base()->GetApi(ORT_API_VERSION);
    if (!p) throw std::runtime_error("libfacehi_ort122.so GetApi(22) 失败");
    return p;
#endif
  }();
  return api;
}

OrtEnv* inner_env() {
  static OrtEnv* env = []() -> OrtEnv* {
    const OrtApi* api = load_inner_api();
    OrtEnv* e = nullptr;
    throw_status(api, api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "facehi-inner-1.22", &e),
                 "CreateEnv");
    return e;
  }();
  return env;
}

OrtSessionOptions* make_so(const OrtApi* api, int intra_op_threads) {
  OrtSessionOptions* so = nullptr;
  throw_status(api, api->CreateSessionOptions(&so), "CreateSessionOptions");
  throw_status(api, api->SetIntraOpNumThreads(so, intra_op_threads), "SetIntraOpNumThreads");
  throw_status(api, api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_BASIC),
               "SetSessionGraphOptimizationLevel");
  return so;
}

}  // namespace

struct InnerSession::Impl {
  const OrtApi* api = nullptr;
  OrtSession* session = nullptr;
};

InnerSession::InnerSession(const void* model_bytes, size_t model_size, int intra_op_threads)
    : impl_(std::make_unique<Impl>()) {
  impl_->api = load_inner_api();
  OrtSessionOptions* so = make_so(impl_->api, intra_op_threads);
  OrtStatus* st = impl_->api->CreateSessionFromArray(inner_env(), model_bytes, model_size, so,
                                                     &impl_->session);
  impl_->api->ReleaseSessionOptions(so);
  throw_status(impl_->api, st, "CreateSessionFromArray");
}

InnerSession::InnerSession(const std::string& model_path, int intra_op_threads)
    : impl_(std::make_unique<Impl>()) {
  impl_->api = load_inner_api();
  OrtSessionOptions* so = make_so(impl_->api, intra_op_threads);
#ifdef _WIN32
  std::wstring w = std::filesystem::u8path(model_path).wstring();
  OrtStatus* st = impl_->api->CreateSession(inner_env(), w.c_str(), so, &impl_->session);
#else
  OrtStatus* st = impl_->api->CreateSession(inner_env(), model_path.c_str(), so, &impl_->session);
#endif
  impl_->api->ReleaseSessionOptions(so);
  throw_status(impl_->api, st, "CreateSession");
}

InnerSession::~InnerSession() {
  if (impl_ && impl_->session && impl_->api) impl_->api->ReleaseSession(impl_->session);
}

std::vector<InnerSession::Tensor> InnerSession::run(const float* input, size_t input_count,
                                                    const int64_t* shape, size_t rank,
                                                    const char* input_name,
                                                    const char* const* output_names,
                                                    size_t n_outputs) const {
  const OrtApi* api = impl_->api;
  OrtMemoryInfo* mem = nullptr;
  throw_status(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem),
               "CreateCpuMemoryInfo");
  OrtValue* in_val = nullptr;
  OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
      mem, const_cast<float*>(input), input_count * sizeof(float), shape, rank,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_val);
  api->ReleaseMemoryInfo(mem);
  throw_status(api, st, "CreateTensorWithDataAsOrtValue");

  std::vector<OrtValue*> outs(n_outputs, nullptr);
  const OrtValue* inputs[1] = {in_val};
  st = api->Run(impl_->session, nullptr, &input_name, inputs, 1, output_names, n_outputs,
                outs.data());
  api->ReleaseValue(in_val);
  throw_status(api, st, "SessionRun");

  std::vector<Tensor> result(n_outputs);
  for (size_t i = 0; i < n_outputs; ++i) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    throw_status(api, api->GetTensorTypeAndShape(outs[i], &info), "GetTensorTypeAndShape");
    size_t dim_count = 0;
    throw_status(api, api->GetDimensionsCount(info, &dim_count), "GetDimensionsCount");
    result[i].shape.resize(dim_count);
    throw_status(api, api->GetDimensions(info, result[i].shape.data(), dim_count),
                 "GetDimensions");
    size_t elem = 0;
    throw_status(api, api->GetTensorShapeElementCount(info, &elem), "GetTensorShapeElementCount");
    api->ReleaseTensorTypeAndShapeInfo(info);
    float* data = nullptr;
    throw_status(api, api->GetTensorMutableData(outs[i], reinterpret_cast<void**>(&data)),
                 "GetTensorMutableData");
    result[i].data.assign(data, data + elem);
    api->ReleaseValue(outs[i]);
  }
  return result;
}

}  // namespace facehi
