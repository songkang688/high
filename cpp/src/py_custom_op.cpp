// libhigh_removal_pyops：ai.high:HighlightRemoval 的「精确内核」蹦床。
//
// 与 libhigh_removal_ops.so（C++ 快速内核）注册完全相同的算子签名，因此同一个
// models/high_removal.onnx 文件不需要任何改动——注册哪个库就用哪个内核。
// 本库不包含第二套算法实现：Compute 时通过 CPython C API 调用宿主进程内的
// highlight_removal.exact_kernel.run_from_buffer，即仓库原始 Python 流水线
// （MediaPipe FaceLandmarker + pip OpenCV process_image）本体，所以
// session.run 输出与直接调用 process_image 逐位相同。
//
// 约束（如实声明）：
//   - 宿主必须是 Python 进程（Python onnxruntime 的 register_custom_ops_library）。
//     本库刻意不链接 libpython（Py_LIMITED_API 稳定 ABI，符号由宿主解释器提供），
//     纯 C++ 宿主 dlopen 时会因符号缺失直接失败——那种场景请用 libhigh_removal_ops.so。
//   - 需要本仓库的 highlight_removal 包与 models/face_landmarker.task 可被导入：
//     默认按本 .so 自身位置推导仓库根（cpp/build/ 的上两级），也可用环境变量
//     HIGH_PY_KERNEL_PATH 显式指定仓库根目录。
#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000
#include <Python.h>

#include <onnxruntime_cxx_api.h>

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr const char* kDomain = "ai.high";
constexpr const char* kOpName = "HighlightRemoval";
constexpr const char* kAttrConfigYaml = "config_yaml";
constexpr const char* kPyModule = "highlight_removal.exact_kernel";
constexpr const char* kPyFunc = "run_from_buffer";
constexpr const char* kEnvKernelPath = "HIGH_PY_KERNEL_PATH";

std::string readBytesAttribute(const Ort::ConstKernelInfo& info, const char* name) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::Value value = info.GetTensorAttribute(name, alloc);
    const size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* data = value.GetTensorData<uint8_t>();
    return std::string(reinterpret_cast<const char*>(data), count);
}

OrtStatusPtr makeStatus(OrtErrorCode code, const std::string& message) {
    return Ort::GetApi().CreateStatus(code, message.c_str());
}

// 借助 GIL 已持有的前提，把当前 Python 异常格式化为字符串并清除。
std::string fetchPyError() {
    PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);
    std::string out = "Python 精确内核执行失败";
    if (value != nullptr) {
        if (PyObject* str = PyObject_Str(value)) {
            if (PyObject* utf8 = PyUnicode_AsUTF8String(str)) {
                out += ": ";
                out.append(PyBytes_AsString(utf8), static_cast<size_t>(PyBytes_Size(utf8)));
                Py_DecRef(utf8);
            }
            Py_DecRef(str);
        }
    }
    if (trace != nullptr) {  // 尽力附上 traceback，便于定位
        if (PyObject* tbMod = PyImport_ImportModule("traceback")) {
            if (PyObject* fmt = PyObject_GetAttrString(tbMod, "format_exception")) {
                if (PyObject* lines = PyObject_CallFunctionObjArgs(fmt, type, value, trace, nullptr)) {
                    if (PyObject* empty = PyUnicode_FromString("")) {
                        if (PyObject* joined = PyObject_CallMethod(empty, "join", "O", lines)) {
                            if (PyObject* utf8 = PyUnicode_AsUTF8String(joined)) {
                                out += "\n";
                                out.append(PyBytes_AsString(utf8), static_cast<size_t>(PyBytes_Size(utf8)));
                                Py_DecRef(utf8);
                            }
                            Py_DecRef(joined);
                        }
                        Py_DecRef(empty);
                    }
                    Py_DecRef(lines);
                }
                Py_DecRef(fmt);
            }
            Py_DecRef(tbMod);
        }
        PyErr_Clear();
    }
    Py_DecRef(type);
    if (value != nullptr) Py_DecRef(value);
    if (trace != nullptr) Py_DecRef(trace);
    return out;
}

// 由本 .so 自身路径推导仓库根：<repo>/cpp/build/libhigh_removal_pyops.so → <repo>。
// 顺序：环境变量 > 上两级（默认构建位置对应仓库根）> 上一级 > 库所在目录。
std::vector<std::string> candidateRepoRoots() {
    std::vector<std::string> roots;
    if (const char* env = std::getenv(kEnvKernelPath); env != nullptr && env[0] != '\0') {
        roots.emplace_back(env);
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&candidateRepoRoots), &info) != 0 && info.dli_fname != nullptr) {
        std::vector<std::string> ups;
        std::string dir(info.dli_fname);
        for (int up = 0; up < 3; ++up) {  // 库目录、上一级、上两级
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir.resize(slash);
            ups.push_back(dir);
        }
        for (auto it = ups.rbegin(); it != ups.rend(); ++it) roots.push_back(*it);
    }
    return roots;
}

void prependSysPath(const std::string& dir) {
    PyObject* path = PySys_GetObject("path");  // 借用引用
    if (path == nullptr || PyList_Check(path) == 0) return;
    // 已在 sys.path 中则跳过。
    const Py_ssize_t n = PyList_Size(path);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyList_GetItem(path, i);
        if (item != nullptr && PyUnicode_Check(item) != 0) {
            if (PyObject* utf8 = PyUnicode_AsUTF8String(item)) {
                const bool same = dir == std::string(PyBytes_AsString(utf8), static_cast<size_t>(PyBytes_Size(utf8)));
                Py_DecRef(utf8);
                if (same) return;
            }
        }
    }
    if (PyObject* str = PyUnicode_FromString(dir.c_str())) {
        PyList_Insert(path, 0, str);
        Py_DecRef(str);
    }
}

PyObject* importKernelModule() {
    PyObject* mod = PyImport_ImportModule(kPyModule);
    if (mod != nullptr) return mod;
    PyErr_Clear();
    for (const std::string& root : candidateRepoRoots()) {
        prependSysPath(root);
        mod = PyImport_ImportModule(kPyModule);
        if (mod != nullptr) return mod;
        PyErr_Clear();
    }
    PyErr_Format(PyExc_ImportError,
                 "无法导入 %s：请在仓库根目录运行，或设置环境变量 %s=<仓库根目录>",
                 kPyModule, kEnvKernelPath);
    return nullptr;
}

struct GilGuard {
    PyGILState_STATE state;
    GilGuard() : state(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state); }
    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;
};

struct ExactKernel {
    std::string configYaml;

    explicit ExactKernel(const OrtKernelInfo* info) {
        const Ort::ConstKernelInfo kinfo{info};
        configYaml = readBytesAttribute(kinfo, kAttrConfigYaml);
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
        try {
            const Ort::KernelContext ctx{context};
            Ort::ConstValue input = ctx.GetInput(0);
            const std::vector<int64_t> shape = input.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[2] != 3 || shape[0] <= 0 || shape[1] <= 0) {
                return makeStatus(ORT_INVALID_ARGUMENT,
                                  "HighlightRemoval: image 输入必须是 [H, W, 3] 的 uint8 BGR 张量");
            }
            const auto rows = shape[0];
            const auto cols = shape[1];
            const size_t imageBytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * 3;
            const size_t maskBytes = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            const auto* pixels = input.GetTensorData<uint8_t>();

            if (Py_IsInitialized() == 0) {
                return makeStatus(ORT_RUNTIME_EXCEPTION,
                                  "精确内核（libhigh_removal_pyops.so）要求宿主为 Python 进程；"
                                  "纯 C++ 宿主请改用 libhigh_removal_ops.so");
            }

            std::string resultBuf;
            std::string maskBuf;
            {
                const GilGuard gil;
                PyObject* mod = importKernelModule();
                if (mod == nullptr) return makeStatus(ORT_RUNTIME_EXCEPTION, fetchPyError());
                PyObject* fn = PyObject_GetAttrString(mod, kPyFunc);
                Py_DecRef(mod);
                if (fn == nullptr) return makeStatus(ORT_RUNTIME_EXCEPTION, fetchPyError());

                PyObject* ret = PyObject_CallFunction(
                    fn, "y#LLy#",
                    reinterpret_cast<const char*>(pixels), static_cast<Py_ssize_t>(imageBytes),
                    static_cast<long long>(rows), static_cast<long long>(cols),
                    configYaml.data(), static_cast<Py_ssize_t>(configYaml.size()));
                Py_DecRef(fn);
                if (ret == nullptr) return makeStatus(ORT_RUNTIME_EXCEPTION, fetchPyError());

                OrtStatusPtr bad = nullptr;
                PyObject* first = nullptr;
                PyObject* second = nullptr;
                if (PyTuple_Check(ret) == 0 || PyTuple_Size(ret) != 2) {
                    bad = makeStatus(ORT_RUNTIME_EXCEPTION, "精确内核返回值必须是 (result_bytes, mask_bytes) 二元组");
                } else {
                    first = PyTuple_GetItem(ret, 0);   // 借用引用
                    second = PyTuple_GetItem(ret, 1);  // 借用引用
                    if (first == nullptr || second == nullptr || PyBytes_Check(first) == 0 || PyBytes_Check(second) == 0) {
                        bad = makeStatus(ORT_RUNTIME_EXCEPTION, "精确内核返回值必须是两个 bytes 对象");
                    } else if (static_cast<size_t>(PyBytes_Size(first)) != imageBytes ||
                               static_cast<size_t>(PyBytes_Size(second)) != maskBytes) {
                        bad = makeStatus(ORT_RUNTIME_EXCEPTION, "精确内核返回的字节长度与输入分辨率不一致");
                    }
                }
                if (bad != nullptr) {
                    Py_DecRef(ret);
                    return bad;
                }
                resultBuf.assign(PyBytes_AsString(first), imageBytes);
                maskBuf.assign(PyBytes_AsString(second), maskBytes);
                Py_DecRef(ret);
            }

            Ort::UnownedValue result = ctx.GetOutput(0, shape.data(), shape.size());
            std::memcpy(result.GetTensorMutableData<uint8_t>(), resultBuf.data(), imageBytes);
            const std::array<int64_t, 2> maskShape{rows, cols};
            Ort::UnownedValue mask = ctx.GetOutput(1, maskShape.data(), maskShape.size());
            std::memcpy(mask.GetTensorMutableData<uint8_t>(), maskBuf.data(), maskBytes);
            return nullptr;
        } catch (const Ort::Exception& exc) {
            return makeStatus(ORT_RUNTIME_EXCEPTION, exc.what());
        } catch (const std::exception& exc) {
            return makeStatus(ORT_RUNTIME_EXCEPTION, exc.what());
        }
    }
};

struct ExactOp : Ort::CustomOpBase<ExactOp, ExactKernel, /*WithStatus=*/true> {
    OrtStatusPtr CreateKernelV2(const OrtApi& /*api*/, const OrtKernelInfo* info, void** kernel) const noexcept {
        try {
            *kernel = new ExactKernel(info);
            return nullptr;
        } catch (const std::exception& exc) {
            return makeStatus(ORT_RUNTIME_EXCEPTION, exc.what());
        }
    }
    const char* GetName() const { return kOpName; }
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
    size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
};

}  // namespace

extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* apiBase) {
    const OrtApi* api = apiBase->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        std::cerr << "[high_removal_pyops] 宿主 ONNX Runtime 不支持 API 版本 " << ORT_API_VERSION
                  << "，请使用 >= 1.22 的 onnxruntime\n";
        std::abort();
    }
    Ort::InitApi(api);

    static ExactOp op;  // 生命周期需覆盖所有会话
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
    static std::vector<OrtCustomOpDomain*> keepAlive;
    static std::mutex keepAliveMutex;
    {
        const std::lock_guard<std::mutex> lock(keepAliveMutex);
        keepAlive.push_back(domain);
    }
    return nullptr;
}
