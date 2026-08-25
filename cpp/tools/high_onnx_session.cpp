// 单 ONNX 模型（models/high_removal.onnx，ai.high:HighlightRemoval 自定义算子）的最小 C++ 调用示例：
//   1. RegisterCustomOpsLibrary 加载 libhigh_removal_ops.so；
//   2. Ort::Session 打开 high_removal.onnx；
//   3. 喂入 uint8 [H, W, 3] BGR 张量，取回 result / hard_mask。
//
// 用法：
//   high_onnx_session --model models/high_removal.onnx --ops cpp/build/libhigh_removal_ops.so \
//                     --input data/1.png --output out.png [--mask mask.png]
#include <onnxruntime_cxx_api.h>

#include <array>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace {

void printUsage(const char* prog) {
    std::cerr << "用法: " << prog << " --model <high_removal.onnx> --ops <libhigh_removal_ops.so>"
              << " --input <图片> --output <输出图片> [--mask <掩码输出>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string modelPath = "models/high_removal.onnx";
    std::string opsPath = "cpp/build/libhigh_removal_ops.so";
    std::string inputPath, outputPath, maskPath;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[错误] " << name << " 缺少参数值\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            modelPath = next("--model");
        } else if (arg == "--ops") {
            opsPath = next("--ops");
        } else if (arg == "--input" || arg == "-i") {
            inputPath = next("--input");
        } else if (arg == "--output" || arg == "-o") {
            outputPath = next("--output");
        } else if (arg == "--mask") {
            maskPath = next("--mask");
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[错误] 未知参数: " << arg << "\n";
            printUsage(argv[0]);
            return 2;
        }
    }
    if (inputPath.empty() || outputPath.empty()) {
        printUsage(argv[0]);
        return 2;
    }

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[错误] 图片读取失败: " << inputPath << "\n";
        return 2;
    }
    if (!image.isContinuous()) image = image.clone();

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "high_onnx_session");
        Ort::SessionOptions sessionOptions;
        sessionOptions.RegisterCustomOpsLibrary(opsPath.c_str());
        Ort::Session session(env, modelPath.c_str(), sessionOptions);

        const std::array<int64_t, 3> shape{image.rows, image.cols, 3};
        const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<uint8_t>(
            mem, image.data, image.total() * 3, shape.data(), shape.size());

        const char* inputNames[] = {"image"};
        const char* outputNames[] = {"result", "hard_mask"};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 2);

        const auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const cv::Mat result(static_cast<int>(outShape[0]), static_cast<int>(outShape[1]), CV_8UC3,
                             outputs[0].GetTensorMutableData<uint8_t>());
        if (!cv::imwrite(outputPath, result)) {
            std::cerr << "[错误] 结果保存失败: " << outputPath << "\n";
            return 5;
        }
        if (!maskPath.empty()) {
            const cv::Mat mask(static_cast<int>(outShape[0]), static_cast<int>(outShape[1]), CV_8UC1,
                               outputs[1].GetTensorMutableData<uint8_t>());
            cv::imwrite(maskPath, mask);
        }
        std::cout << outputPath << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[错误] " << exc.what() << "\n";
        return 5;
    }
}
