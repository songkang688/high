// facehi_cpp：C++ 版去高光命令行入口，行为对齐 cli_process.py「常用模式」默认。
//
// 用法：
//   facehi_cpp -i data/1.png -o out.png
//   facehi_cpp -i data/ -o out_dir/                 # 批量
//   facehi_cpp -i in.png --mode 最高质量模式
//   facehi_cpp -i in.png --landmarks 1_landmarks.txt --dump-dir dbg/   # 黄金对照
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "facehi/config.hpp"
#include "facehi/pipeline.hpp"
#include "facehi/utils.hpp"

namespace fs = std::filesystem;
using namespace facehi;

namespace {

std::string find_repo_root(const std::string& hint) {
  std::vector<std::string> candidates;
  if (!hint.empty()) candidates.push_back(hint);
  candidates.push_back(".");
  candidates.push_back("..");
  candidates.push_back("../..");
  for (const auto& c : candidates)
    if (fs::exists(fs::path(c) / "configs" / "default.yaml")) return fs::absolute(c).string();
  return "";
}

cv::Mat load_landmarks_txt(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    n = std::stoi(line);
    break;
  }
  if (n <= 0) return {};
  cv::Mat lm(n, 3, CV_32F);
  for (int i = 0; i < n; ++i) {
    if (!std::getline(f, line)) return {};
    std::istringstream ss(line);
    double x, y, z;
    ss >> x >> y >> z;
    lm.at<float>(i, 0) = static_cast<float>(x);
    lm.at<float>(i, 1) = static_cast<float>(y);
    lm.at<float>(i, 2) = static_cast<float>(z);
  }
  return lm;
}

bool is_image_file(const fs::path& p) {
  std::string ext = p.extension().string();
  for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp";
}

int process_one(const fs::path& input, const fs::path& output, const Config& cfg,
                const OnnxFaceLandmarker* landmarker, const cv::Mat* injected,
                const std::string& dump_dir) {
  cv::Mat image = read_image_bgr(input.string());
  if (image.empty()) {
    std::cerr << "[失败] " << input.filename().string() << "：图片读取失败" << std::endl;
    return 1;
  }
  PipelineOutput out = process_image(image, cfg, landmarker, injected);
  if (!out.success) {
    std::cerr << "[失败] " << input.filename().string() << "：" << out.status << std::endl;
    return 1;
  }
  if (!write_image(output.string(), out.result_bgr)) {
    std::cerr << "[失败] 结果保存失败：" << output.string() << std::endl;
    return 1;
  }
  std::cout << fs::absolute(output).string() << std::endl;
  if (!out.warnings.empty()) {
    for (const auto& w : out.warnings) std::cerr << "[警告] " << w << std::endl;
  }
  if (!dump_dir.empty()) {
    fs::create_directories(dump_dir);
    std::string stem = input.stem().string();
    write_image((fs::path(dump_dir) / (stem + "_hard.png")).string(), out.highlight_mask);
    write_image((fs::path(dump_dir) / (stem + "_soft.png")).string(), out.soft_mask);
    if (out.regions) {
      write_image((fs::path(dump_dir) / (stem + "_skin.png")).string(), out.regions->masks.at("skin"));
      write_image((fs::path(dump_dir) / (stem + "_protect.png")).string(), out.regions->masks.at("protect"));
      if (out.regions->masks.count("treatable_skin"))
        write_image((fs::path(dump_dir) / (stem + "_treatable.png")).string(),
                    out.regions->masks.at("treatable_skin"));
    }
    std::ofstream mf(fs::path(dump_dir) / (stem + "_metrics.txt"));
    mf << out.metrics_text;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Ort::InitApi();  // facehi_core 以 ORT_API_MANUAL_INIT 编译，独立进程需手动初始化。
  std::string input, output, mode = "常用模式", repo_hint, landmarks_path, dump_dir, models_dir;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "-i" || a == "--input")
      input = next();
    else if (a == "-o" || a == "--output")
      output = next();
    else if (a == "--mode")
      mode = next();
    else if (a == "--repo-root")
      repo_hint = next();
    else if (a == "--landmarks")
      landmarks_path = next();
    else if (a == "--dump-dir")
      dump_dir = next();
    else if (a == "--models-dir")
      models_dir = next();
    else if (a == "-h" || a == "--help") {
      std::cout << "用法：facehi_cpp -i <图片或文件夹> [-o 输出] [--mode 常用模式|高保真模式|最高质量模式]\n"
                   "      [--landmarks lm.txt] [--dump-dir dir] [--models-dir dir] [--repo-root dir]\n";
      return 0;
    }
  }
  if (input.empty()) {
    std::cerr << "[错误] 未提供输入路径，请使用 -i 指定图片或文件夹。" << std::endl;
    return 2;
  }
  std::string root = find_repo_root(repo_hint);
  if (root.empty()) {
    std::cerr << "[错误] 找不到 configs/default.yaml，请用 --repo-root 指定仓库根目录。" << std::endl;
    return 2;
  }
  Config cfg = load_cli_config(root, mode);

  cv::Mat injected;
  if (!landmarks_path.empty()) {
    injected = load_landmarks_txt(landmarks_path);
    if (injected.empty()) {
      std::cerr << "[错误] 关键点文件读取失败：" << landmarks_path << std::endl;
      return 2;
    }
  }

  std::unique_ptr<OnnxFaceLandmarker> landmarker;
  if (injected.empty()) {
    std::string mdir = models_dir.empty() ? root + "/onnx/models" : models_dir;
    std::string det_path = mdir + "/face_detector.onnx";
    std::string lmk_path = mdir + "/face_landmarks_detector.onnx";
    if (!fs::exists(det_path) || !fs::exists(lmk_path)) {
      std::cerr << "[错误] 人脸模型文件缺失。\n       期望位置：" << det_path << "\n"
                << "       可运行 onnx/export_onnx.py 从 models/face_landmarker.task 生成。" << std::endl;
      return 3;
    }
    double min_conf = cfg.face_detection.getd("face_detection_confidence", 0.55);
    double lm_conf = cfg.face_detection.getd("landmark_detection_confidence", 0.50);
    int max_faces = cfg.face_detection.geti("max_faces", 4);
    landmarker = std::make_unique<OnnxFaceLandmarker>(det_path, lmk_path, min_conf, lm_conf, max_faces);
  }

  fs::path in_path(input);
  if (fs::is_directory(in_path)) {
    std::vector<fs::path> images;
    for (const auto& e : fs::directory_iterator(in_path))
      if (e.is_regular_file() && is_image_file(e.path())) images.push_back(e.path());
    std::sort(images.begin(), images.end());
    fs::path out_dir = output.empty() ? in_path : fs::path(output);
    fs::create_directories(out_dir);
    int ok = 0, fail = 0;
    for (const auto& img : images) {
      fs::path out_file = out_dir / (img.stem().string() + "_output" + img.extension().string());
      if (process_one(img, out_file, cfg, landmarker.get(),
                      injected.empty() ? nullptr : &injected, dump_dir) == 0)
        ++ok;
      else
        ++fail;
    }
    std::cerr << "[完成] 成功 " << ok << " 张，失败 " << fail << " 张，共 " << images.size() << " 张"
              << std::endl;
    return fail == 0 ? 0 : (ok == 0 ? 5 : 6);
  }

  fs::path out_file;
  if (output.empty())
    out_file = in_path.parent_path() / (in_path.stem().string() + "_output" + in_path.extension().string());
  else if (fs::is_directory(fs::path(output)))
    out_file = fs::path(output) / (in_path.stem().string() + "_output" + in_path.extension().string());
  else
    out_file = fs::path(output);
  return process_one(in_path, out_file, cfg, landmarker.get(),
                     injected.empty() ? nullptr : &injected, dump_dir);
}
