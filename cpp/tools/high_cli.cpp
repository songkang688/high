// 命令行入口：
//   high_onnx --input data/1.png --output out.png --config configs/default.yaml [--models models/]
// 输出与 Python cli_process.py（常用模式 / default.yaml）对应的去高光结果，纯 CPU。
#include <cstring>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "high_removal/pipeline.hpp"

namespace {

void printUsage(const char* prog) {
    std::cerr << "用法: " << prog << " --input <图片> --output <输出图片> --config <default.yaml>"
              << " [--models <ONNX 模型目录>] [--dump-masks <前缀>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string inputPath, outputPath, configPath, modelDir = "models", dumpPrefix;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[错误] " << name << " 缺少参数值\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--input" || arg == "-i") {
            inputPath = next("--input");
        } else if (arg == "--output" || arg == "-o") {
            outputPath = next("--output");
        } else if (arg == "--config" || arg == "-c") {
            configPath = next("--config");
        } else if (arg == "--models" || arg == "-m") {
            modelDir = next("--models");
        } else if (arg == "--dump-masks") {
            dumpPrefix = next("--dump-masks");
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[错误] 未知参数: " << arg << "\n";
            printUsage(argv[0]);
            return 2;
        }
    }
    if (inputPath.empty() || outputPath.empty() || configPath.empty()) {
        printUsage(argv[0]);
        return 2;
    }

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[错误] 图片读取失败: " << inputPath << "\n";
        return 2;
    }

    hr::Config config;
    try {
        config = hr::loadConfig(configPath);
    } catch (const std::exception& exc) {
        std::cerr << "[错误] 配置加载失败: " << exc.what() << "\n";
        return 2;
    }

    try {
        const int numFaces = static_cast<int>(hr::getF(config.face_detection, {"max_faces"}, 4));
        const float minDet = static_cast<float>(hr::getF(config.face_detection, {"face_detection_confidence"}, 0.55));
        const float minPresence = static_cast<float>(hr::getF(config.face_detection, {"landmark_detection_confidence"}, 0.50));
        hr::FaceLandmarkerOrt landmarker(modelDir, numFaces, minDet, minPresence);

        const hr::PipelineOutput result = hr::processImage(image, config, landmarker);
        for (const auto& warn : result.warnings) std::cerr << "[警告] " << warn << "\n";
        if (!result.success) {
            std::cerr << "[失败] " << result.status << "\n";
            return 5;
        }
        if (!cv::imwrite(outputPath, result.result_bgr)) {
            std::cerr << "[错误] 结果保存失败: " << outputPath << "\n";
            return 5;
        }
        if (!dumpPrefix.empty()) {
            cv::imwrite(dumpPrefix + "_hard_mask.png", result.highlight_mask);
            cv::imwrite(dumpPrefix + "_soft_mask.png", result.soft_mask);
        }
        std::cout << outputPath << "\n";
        std::cerr << "[信息] " << result.status << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[错误] 处理失败: " << exc.what() << "\n";
        return 5;
    }
}
