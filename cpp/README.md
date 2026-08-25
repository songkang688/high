# 证件照去高光 C++ 版（OpenCV + ONNX Runtime，纯 CPU）

Python 推理链路（`highlight_removal/`）的 1:1 移植：

- 人脸 478 关键点：ONNX Runtime 跑 `models/face_detector.onnx` + `models/face_landmarks_detector.onnx`
  （由 `models/face_landmarker.task` 内的原始 TFLite 精确转换，见 `tools/export_landmarker_onnx.py`），
  MediaPipe 图的前后处理（letterbox、BlazeFace anchor 解码、加权 NMS、旋转裁剪、坐标反投影）按官方源码在
  `src/landmarker.cpp` 中重实现，与 MediaPipe 输出的关键点偏差 < 0.25px；
- 区域划分 / 肤色掩码 / 高光检测 / 去高光修复：OpenCV C++ 按相同公式与参数移植
  （LAB 逐通道 TELEA inpaint 0.72/0.28 混合、双边滤波 d=11 σ=45/45、Canny 60/130、区域分位数自适应阈值等）；
- 配置直接读 `configs/default.yaml`（yaml-cpp），缺省值与 Python 代码内默认值一致。

实测与 Python 输出的一致性（17 张样例图）：平均 MAE 0.0028/255、平均 PSNR 72.6 dB、
高光硬掩码平均 IoU 0.9996，详见 [`tools/PARITY_RESULTS.md`](../tools/PARITY_RESULTS.md)。
为什么不是位级 100% 一致、以及为什么整条流水线无法放进一个标准 ONNX 图，见
[`docs/ONNX_CPP_PLAN.md`](../docs/ONNX_CPP_PLAN.md)。

## 依赖

| 依赖 | 说明 |
|------|------|
| CMake ≥ 3.16、g++（C++17） | Ubuntu：`sudo apt install cmake g++` |
| OpenCV（core/imgproc/imgcodecs/photo） | Ubuntu：`sudo apt install libopencv-dev` |
| yaml-cpp | Ubuntu：`sudo apt install libyaml-cpp-dev` |
| ONNX Runtime（CPU 预编译包） | [官方 Releases](https://github.com/microsoft/onnxruntime/releases) 下载 `onnxruntime-linux-x64-<版本>.tgz` 解压即可 |

示例（Linux x64）：

```bash
sudo apt install cmake g++ libopencv-dev libyaml-cpp-dev
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz   # 假设解压到 /opt/ort/
```

## 构建

```bash
cd cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++ \
      -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 ..
make -j$(nproc)
```

说明：若系统默认 `c++` 指向 clang 且缺少对应 libstdc++ 开发文件，请像上面一样显式指定 `g++`。

## 运行

在仓库根目录：

```bash
LD_LIBRARY_PATH=/opt/ort/onnxruntime-linux-x64-1.22.0/lib \
  ./cpp/build/high_onnx --input data/1.png --output out.png --config configs/default.yaml
```

参数：

- `--input/-i`、`--output/-o`、`--config/-c`：必填；
- `--models/-m`：ONNX 模型目录，默认 `models`；
- `--dump-masks 前缀`：另存高光硬/软掩码（调试用）。

纯 CPU 运行，无任何 GPU 依赖（与 Python 侧 `FORCE_CPU_ONLY=True` 一致）。

## 与 Python 版的对齐范围

- 完整支持 `configs/default.yaml` 默认路径：`process_scale=compromise`（人脸 1/4 + 高光 1/2 + 原图修复）、
  `simple_protect_mode=true`、`mode=混合`、ROI 修复；`key_region_detection_mode`（九区域模式）、
  `保真`/`强修复` 模式和数值型 `process_scale` 也已按相同公式移植。
- 未移植（默认配置不会走到）：`enable_alignment=true` 的轻量对齐分支、`refine_upsampled_masks=true`
  的 ROI 全分辨率重检测分支（该开关打开时按原分辨率整图检测处理，结果正确但比 Python 的 ROI 重检测慢）、
  OpenCV Haar/LBF 兜底检测器（模型缺失时直接报错，不降级）、质量审计文本与调试可视化。

## 对拍

```bash
python tools/compare_python_cpp.py            # 全部 data/*.png
python tools/compare_python_cpp.py --images data/1.png data/2.png
```
