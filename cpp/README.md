# facehi_cpp：C++ 版面部去高光流水线

本目录是 Python 包 `highlight_removal/` 的 C++ 移植版，目标是与
`cli_process.py`「常用模式」输出**像素级对齐**（已达成：注入同一关键点时
17/17 黄金样本 max abs diff = 0，见 `tests/golden_report.md`）。神经网络部分
（MediaPipe FaceLandmarker 的人脸检测 + 478 点关键点）使用从
`models/face_landmarker.task` 拆出并转换的 ONNX 模型，经 ONNX Runtime 推理；
其余全部为 OpenCV C++ 复刻。

> **只想要一个 .onnx 模型来调用？** 见 [`onnx/README.md`](../onnx/README.md)：
> `onnx/models/facehi.onnx`（单节点自定义算子封装整条流水线，内嵌子模型与配置）
> + 本目录构建出的 `libfacehi_custom_ops.so` 即是全部交付物。

## 为什么不能用「标准算子拼出的纯 ONNX 图」实现 100% 效果

ONNX 只能表达张量计算图。本项目的主体是**传统 CV 流水线**，包含大量无法用
标准 ONNX 算子表达（或表达后无法保持像素级一致）的步骤：

| Python 步骤 | 为什么进不了单个 ONNX |
| --- | --- |
| MediaPipe Tasks 运行时 | `.task` 内是 TFLite 模型 + 图配置，anchor 解码/加权 NMS/ROI 旋转裁剪等后处理在运行时 C++ 里，不在模型图内 |
| `cv2.connectedComponentsWithStats` | 连通域标记是数据相关的迭代算法，无对应 ONNX 算子 |
| `cv2.inpaint`（Telea） | 快速行进法，基于堆的逐像素迭代，无 ONNX 算子 |
| `cv2.bilateralFilter` / Haar 级联 | 无标准算子 |
| 区域分位数自适应阈值、从核心向晕区生长、连通域面积截断 | 依赖动态形状与数据相关控制流 |
| 质量审计的统计与中文警告 | 字符串逻辑，非张量计算 |

因此正确形态是：算法主体用 C++ 复刻（本目录），再按 ORT 官方
「自定义算子包装外部流水线」把整条链路封进**单个 facehi.onnx**
（见 `onnx/README.md`）。等价性用 `data/*.png` 黄金对照验证
（见 `tests/golden_report.md`：注入同一关键点时，17 张样本全部
mask IoU = 1.0 且**结果图 max abs diff = 0，位级一致**）。

## 目录结构

```
cpp/
├── CMakeLists.txt
├── include/facehi/          # 头文件（与 Python 模块一一对应）
│   ├── common.hpp           # 参数表 / Python 取整 / numpy 分位数
│   ├── config.hpp           # YAML 配置加载（configs/default.yaml + 预设合并）
│   ├── face_landmarks.hpp   # 468/478 语义索引与几何（face_landmarks.py）
│   ├── face_detect.hpp      # ONNX 人脸检测+关键点（face_detect.py）
│   ├── face_regions.hpp     # 皮肤估计与分区（skin_mask.py / face_regions.py）
│   ├── highlight_detect.hpp # 高光检测（highlight_detect.py）
│   ├── remove_highlight.hpp # 高光修复（remove_highlight.py）
│   ├── quality_metrics.hpp  # 质量审计（quality_metrics.py）
│   ├── pipeline.hpp         # 总控 process_image()（pipeline.py）
│   └── utils.hpp            # mask/缩放/读写工具（utils.py）
├── src/                     # 对应实现
│   └── custom_op.cpp        # ORT 自定义算子 ai.facehi:HighlightRemoval
├── tools/facehi_cli.cpp     # CLI：facehi_cpp -i data/1.png -o out.png
├── tools/facehi_onnx_cli.cpp# CLI：单一 facehi.onnx 入口的最小集成示例
├── tests/
│   ├── export_golden.py     # 从 Python 侧导出黄金对照数据
│   ├── compare_golden.py    # C++ vs Python 逐像素对照
│   └── golden_report.md     # 17 张样本的完整数值报告
└── golden/                  # 黄金样本(仓库内保留 1、13 两组，其余可重新生成)
```

## 依赖与构建

- CMake ≥ 3.16、GCC/Clang（C++17）
- OpenCV **4.14.0**（与 Python 侧 `opencv-contrib-python` 同版本，且须
  **WITH_IPP=ON**（PyPI wheel 即如此构建），否则无法位级对齐；
  需要模块 core/imgproc/imgcodecs/photo/objdetect。构建 facehi_custom_ops
  共享库时建议静态库：`-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON`）
- ONNX Runtime（官方预编译包，验证版本 1.29.0）
- yaml-cpp（`apt install libyaml-cpp-dev`）

```bash
# 1) 先生成 ONNX 模型（若 onnx/models/ 下已存在可跳过）
.venv/bin/python onnx/export_onnx.py

# 2) 构建
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DOpenCV_DIR=<opencv安装目录>/lib/cmake/opencv4 \
      -DORT_ROOT=<onnxruntime解压目录>
cmake --build cpp/build -j
# 产物：facehi_cpp（直调 CLI）、libfacehi_custom_ops.so（ORT 自定义算子库）、
#       facehi_onnx_cli（单一 facehi.onnx 入口 CLI）
```

> 数值对齐关键编译选项已写入 CMakeLists：`-ffp-contract=off -fno-fast-math`，
> 禁止 FMA 收缩；`remove_highlight.cpp` 并按 numpy 2（NEP 50）的标量提升语义
> 区分 float32/float64 计算路径，保证与 Python 逐步舍入一致。

## 使用

```bash
# 单张（默认输出 <名字>_output.png 到同目录）
cpp/build/facehi_cpp -i data/1.png -o out.png

# 批量整个文件夹
cpp/build/facehi_cpp -i data/ -o out_dir/

# 模式与 cli_process.py 相同：常用模式（默认）/ 高保真模式 / 最高质量模式
cpp/build/facehi_cpp -i data/1.png --mode 最高质量模式

# 黄金对照调试：注入 Python 导出的关键点 + 转储中间 mask
cpp/build/facehi_cpp -i cpp/golden/1_input.png -o /tmp/1.png \
    --landmarks cpp/golden/1_landmarks.txt --dump-dir /tmp/dbg
```

模型与配置默认按仓库布局自动定位（`onnx/models/*.onnx`、`configs/*.yaml`、
`models/haarcascade*.xml` 由 CLI 从可执行文件位置向上查找，也可用
`--repo-root` / `--models-dir` 显式指定）。

## C++ API（单一入口）

```cpp
#include "facehi/pipeline.hpp"
#include "facehi/config.hpp"
#include "facehi/face_detect.hpp"

facehi::Config cfg = facehi::load_cli_config(repo_root, "常用模式");
facehi::OnnxFaceLandmarker lmk(det_onnx, lmk_onnx);   // 可复用，线程内安全
facehi::PipelineOutput out = facehi::process_image(bgr, cfg, &lmk);
// out.result_bgr / out.highlight_mask / out.soft_mask / out.warnings / out.metrics_text
```

## 与 Python 的对照方法

```bash
.venv/bin/python cpp/tests/export_golden.py data/*.png -o cpp/golden   # 导出黄金
.venv/bin/python cpp/tests/compare_golden.py --names 1 2 3 ... 17      # 完整链路
.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks ... # 只验 CV 流水线
.venv/bin/python onnx/tests/compare_facehi_onnx.py --names 1 2 3 ... 17 # 单一 ONNX 三路对照
```

完整数字见 [`tests/golden_report.md`](tests/golden_report.md) 与
[`onnx/tests/onnx_report.md`](../onnx/tests/onnx_report.md)。
