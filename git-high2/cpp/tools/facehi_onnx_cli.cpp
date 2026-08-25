// facehi_onnx_cli：单一 ONNX 模型入口的最小集成示例。
//
// 只依赖两个交付文件：
//   onnx/models/facehi.onnx        （唯一对外模型，内嵌子模型与配置）
//   libfacehi_custom_ops.so        （ORT 自定义算子库）
//
// 用法：
//   facehi_onnx_cli -i data/1.png -o out.png
//   facehi_onnx_cli -i in.png -o out.png --model facehi.onnx --ops-lib libfacehi_custom_ops.so
//   facehi_onnx_cli -i in.png -o out.png --mask mask.png
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace {

std::string find_upwards(const std::string& rel, int levels = 4) {
  fs::path base = fs::current_path();
  for (int i = 0; i <= levels; ++i) {
    fs::path cand = base / rel;
    if (fs::exists(cand)) return fs::absolute(cand).string();
    base = base.parent_path();
  }
  return "";
}

cv::Mat imread_bgr(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return cv::imdecode(buf, cv::IMREAD_COLOR);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, output, model_path, ops_lib, mask_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "-i" || a == "--input")
      input = next();
    else if (a == "-o" || a == "--output")
      output = next();
    else if (a == "--model")
      model_path = next();
    else if (a == "--ops-lib")
      ops_lib = next();
    else if (a == "--mask")
      mask_path = next();
    else if (a == "-h" || a == "--help") {
      std::cout << "用法：facehi_onnx_cli -i in.png -o out.png [--model facehi.onnx]"
                   " [--ops-lib libfacehi_custom_ops.so] [--mask mask.png]\n";
      return 0;
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "[错误] 需要 -i 输入与 -o 输出。" << std::endl;
    return 2;
  }
  if (model_path.empty()) model_path = find_upwards("onnx/models/facehi.onnx");
  if (ops_lib.empty()) {
    // 默认在可执行文件同目录查找。
    fs::path self_dir = fs::canonical("/proc/self/exe").parent_path();
    fs::path cand = self_dir / "libfacehi_custom_ops.so";
    if (fs::exists(cand)) ops_lib = cand.string();
  }
  if (model_path.empty() || ops_lib.empty()) {
    std::cerr << "[错误] 找不到 facehi.onnx 或 libfacehi_custom_ops.so，请用 --model/--ops-lib 指定。"
              << std::endl;
    return 2;
  }

  cv::Mat bgr = imread_bgr(input);
  if (bgr.empty()) {
    std::cerr << "[错误] 图片读取失败：" << input << std::endl;
    return 1;
  }
  if (!bgr.isContinuous()) bgr = bgr.clone();

  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "facehi_onnx_cli"};
  Ort::SessionOptions so;
  so.RegisterCustomOpsLibrary(ops_lib.c_str());
  Ort::Session session(env, model_path.c_str(), so);

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> dims = {bgr.rows, bgr.cols, 3};
  Ort::Value image_in = Ort::Value::CreateTensor<uint8_t>(
      mem, bgr.data, static_cast<size_t>(bgr.total()) * 3, dims.data(), dims.size());

  const char* input_names[] = {"image"};
  const char* output_names[] = {"result", "highlight_mask"};
  auto outs = session.Run(Ort::RunOptions{}, input_names, &image_in, 1, output_names, 2);

  cv::Mat result(bgr.rows, bgr.cols, CV_8UC3,
                 const_cast<uint8_t*>(outs[0].GetTensorData<uint8_t>()));
  if (!cv::imwrite(output, result)) {
    std::cerr << "[错误] 结果保存失败：" << output << std::endl;
    return 1;
  }
  std::cout << fs::absolute(output).string() << std::endl;

  if (!mask_path.empty()) {
    cv::Mat mask(bgr.rows, bgr.cols, CV_8U,
                 const_cast<uint8_t*>(outs[1].GetTensorData<uint8_t>()));
    cv::imwrite(mask_path, mask);
  }
  return 0;
}
