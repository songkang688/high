#pragma once

// 嵌套 ORT 会话：dlopen 同目录的独立 ORT 1.22 副本，避免 Python 宿主
// 已经占用 LoggingManager::Default 导致 CreateEnv 失败。

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace facehi {

class InnerSession {
 public:
  InnerSession(const void* model_bytes, size_t model_size, int intra_op_threads);
  InnerSession(const std::string& model_path, int intra_op_threads);
  ~InnerSession();

  InnerSession(const InnerSession&) = delete;
  InnerSession& operator=(const InnerSession&) = delete;

  // 单输入 float NHWC → 多个输出，返回每个输出的 float 缓冲与形状。
  struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
    const float* ptr() const { return data.data(); }
  };
  std::vector<Tensor> run(const float* input, size_t input_count, const int64_t* shape,
                          size_t rank, const char* input_name,
                          const char* const* output_names, size_t n_outputs) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace facehi
