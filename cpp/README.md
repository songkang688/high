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

本目录现在构建三个产物：

| 目标 | 说明 |
|------|------|
| `libhigh_removal_ops.so` | **主要交付物**：ORT 自定义算子库，`models/high_removal.onnx` 中 `ai.high:HighlightRemoval` 算子的内核（微软官方 custom-op wrapper 方案）。只导出 `RegisterCustomOps`，不链接 libonnxruntime（OrtApi 由宿主传入），可被 Python `onnxruntime`（>= 1.22）或任意 C++ ORT 应用加载 |
| `high_onnx_session` | 单 ONNX 会话的最小 C++ 示例（RegisterCustomOpsLibrary + Ort::Session + Run） |
| `high_onnx` | 旧的直接调用 C++ 库的 CLI（保留，输出与单 ONNX 会话逐位相同） |

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

## 运行（推荐）：单 ONNX 模型 + 自定义算子库

`models/high_removal.onnx` 已内嵌两个人脸网络与 `configs/default.yaml`，用户只需要
**一个 onnx 文件 + 一个 .so**（若模型或配置有改动，用
`python tools/export_high_removal_onnx.py` 重新生成 onnx）。

Python（只依赖 `onnxruntime` 与 `opencv-python`，不依赖本仓库其它代码）：

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("cpp/build/libhigh_removal_ops.so")
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
result = sess.run(["result"], {"image": cv2.imread("data/1.png")})[0]
```

或直接用现成脚本 / C++ 示例（在仓库根目录）：

```bash
python tools/run_high_onnx.py --input data/1.png --output out.png

LD_LIBRARY_PATH=/opt/ort/onnxruntime-linux-x64-1.22.0/lib \
  ./cpp/build/high_onnx_session --model models/high_removal.onnx \
  --ops cpp/build/libhigh_removal_ops.so --input data/1.png --output out.png
```

接口：输入 `image` uint8 `[H, W, 3]`（BGR，与 `cv2.imread` 一致，H/W 动态）；
输出 `result` uint8 `[H, W, 3]`（去高光结果）、`hard_mask` uint8 `[H, W]`（高光硬掩码）。

说明：`.so` 是该自定义算子的内核实现，ONNX Runtime 执行自定义算子必须先注册它
（与「运行任何 onnx 都要装 onnxruntime」同理）；运行期依赖系统 OpenCV 与 yaml-cpp
（`sudo apt install libopencv-dev libyaml-cpp-dev` 装出来的运行库即可）。

## 运行（旧 CLI，直接调 C++ 库）

```bash
LD_LIBRARY_PATH=/opt/ort/onnxruntime-linux-x64-1.22.0/lib \
  ./cpp/build/high_onnx --input data/1.png --output out.png --config configs/default.yaml
```

参数：

- `--input/-i`、`--output/-o`、`--config/-c`：必填；
- `--models/-m`：ONNX 模型目录，默认 `models`；
- `--dump-masks 前缀`：另存高光硬/软掩码（调试用）。

两条路径输出逐位相同；均为纯 CPU 运行，无任何 GPU 依赖（与 Python 侧 `FORCE_CPU_ONLY=True` 一致）。

## 与 Python 版的对齐范围

- 完整支持 `configs/default.yaml` 默认路径：`process_scale=compromise`（人脸 1/4 + 高光 1/2 + 原图修复）、
  `simple_protect_mode=true`、`mode=混合`、ROI 修复；`key_region_detection_mode`（九区域模式）、
  `保真`/`强修复` 模式和数值型 `process_scale` 也已按相同公式移植。
- 未移植（默认配置不会走到）：`enable_alignment=true` 的轻量对齐分支、`refine_upsampled_masks=true`
  的 ROI 全分辨率重检测分支（该开关打开时按原分辨率整图检测处理，结果正确但比 Python 的 ROI 重检测慢）、
  OpenCV Haar/LBF 兜底检测器（模型缺失时直接报错，不降级）、质量审计文本与调试可视化。

## 对拍

```bash
python tools/compare_python_cpp.py            # Python vs 旧 CLI，全部 data/*.png
python tools/compare_python_onnx_session.py   # Python vs 单 ONNX 会话，全部 data/*.png
python tools/compare_python_cpp.py --images data/1.png data/2.png
```
