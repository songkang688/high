# git-high1 · 证件照去高光便携包（最佳效果 ONNX + 前端 + 调用脚本）

本目录是 `high` 仓库的**自包含可运行子集**：单文件模型 `models/high_removal.onnx`
（内嵌两个人脸网络与 `configs/default.yaml`）+ 工作室前端 `app_studio.py` +
命令行调用脚本 `tools/run_high_onnx.py`，以及运行所需的算子库与示例图。

在本机（cloud agent VM）上已安装到以下位置，可直接取用：

| 位置 | 说明 |
|------|------|
| `/home/ubuntu/git-high1/` | 主安装位置（本 README 所描述的完整包） |
| `/opt/cursor/artifacts/git-high1/` | Cursor 产物目录副本（→ `/cursor/stores/self/artifacts/git-high1`） |
| `~/.local/share/git-high1` | 指向 `/home/ubuntu/git-high1` 的软链接 |

在仓库里用 `bash tools/export_git_high1.sh [目标目录]` 可随时重新生成本包
（缺 `.so` 时会自动重新编译，见下文）。

## 目录结构

```
git-high1/
  README.md                # 本文件
  app_studio.py            # 前端：Python / ONNX / 对比 三种引擎的工作室页面
  tools/run_high_onnx.py   # 命令行调用入口（session.run 示例，零仓库依赖）
  models/high_removal.onnx       # 单文件模型（内嵌人脸网络 + 默认配置，5.1 MB）
  models/face_landmarker.task    # MediaPipe 人脸模型（精确内核 / Python 引擎需要）
  configs/*.yaml           # default.yaml 与工作室用到的预设
  highlight_removal/       # Python 包（精确内核经它调用原始流水线 process_image）
  cpp/build/libhigh_removal_pyops.so  # 精确内核（效果最好：与 Python 逐位一致）
  cpp/build/libhigh_removal_ops.so    # C++ 快速内核（无 Python/MediaPipe 依赖）
  data/                    # 17 张示例图（前端下拉框直接可选）
```

## 依赖

```bash
pip install onnxruntime opencv-contrib-python mediapipe gradio numpy PyYAML Pillow scikit-image
```

只用 **C++ 快速内核**跑命令行时最小依赖仅为 `onnxruntime opencv-python numpy`；
精确内核 / Python 引擎 / 前端 需要完整一行（mediapipe、gradio 等）。

## 启动前端

```bash
cd /home/ubuntu/git-high1
python app_studio.py                    # http://127.0.0.1:7861
HIGH_STUDIO_HOST=0.0.0.0 HIGH_STUDIO_PORT=8080 python app_studio.py   # 改地址/端口
HIGH_ONNX_KERNEL=cpp python app_studio.py                             # ONNX 侧改用 C++ 快速内核
```

页面提供三种引擎：**Python**（MediaPipe + OpenCV 原始链路）、**ONNX**（单文件模型 +
自定义算子库）、**对比**（同图并排跑两个引擎，显示差异热力图与 MAE/PSNR/IoU 指标）。

## 命令行调用

```bash
cd /home/ubuntu/git-high1
python tools/run_high_onnx.py -i data/1.png -o out.png                # 精确内核（默认）
python tools/run_high_onnx.py --kernel cpp -i data/1.png -o out.png   # C++ 快速内核
python tools/run_high_onnx.py -i data/1.png -o out.png --mask mask.png  # 另存高光硬掩码
```

## session.run 直接调用（核心就这几行）

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("cpp/build/libhigh_removal_pyops.so")  # 或 libhigh_removal_ops.so
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
result, hard_mask = sess.run(["result", "hard_mask"], {"image": cv2.imread("data/1.png")})
```

接口：输入 `image` uint8 `[H, W, 3]`（BGR，`cv2.imread` 原样，H/W 动态）；
输出 `result` uint8 `[H, W, 3]`（去高光结果）、`hard_mask` uint8 `[H, W]`（高光硬掩码）。

## 精确内核 vs C++ 快速内核

同一个 `high_removal.onnx` 支持两个可互换的内核库——注册哪个 `.so` 就用哪个内核，
模型文件不需要任何改动：

| 内核 | 库文件 | 效果 | 依赖 |
|------|--------|------|------|
| **精确内核（默认，效果最好）** | `libhigh_removal_pyops.so` | 与 Python 流水线 `process_image` **逐位相同**（17/17 张 `np.array_equal`，MAE 恒为 0） | 宿主必须是 Python 进程；需要本目录的 `highlight_removal/` 包、`models/face_landmarker.task` 与 mediapipe |
| C++ 快速内核 | `libhigh_removal_ops.so` | 完整 C++ 移植，存在亚像素级浮点尾差（17 张实测平均 MAE ≈ 0.003/255，PSNR ≈ 72 dB，掩码 IoU ≈ 0.9996） | 无 Python/MediaPipe 依赖，可被纯 C++ ORT 宿主加载；运行期需系统 OpenCV 与 yaml-cpp 运行库 |

精确内核按 `.so` 自身位置自动定位本目录（`cpp/build/` 的上两级）；从其他目录导入时
可设 `HIGH_PY_KERNEL_PATH=/home/ubuntu/git-high1` 显式指定。`HIGH_OPS_LIB=/path/to/lib.so`
可直接指定算子库文件（优先级最高）。

## 重新编译 `.so` / 重新生成本包

两个库由仓库 `cpp/` 的同一次 cmake/make 产出（需要
`sudo apt install cmake g++ libopencv-dev libyaml-cpp-dev python3-dev` 与
[onnxruntime 官方预编译包](https://github.com/microsoft/onnxruntime/releases)）。
在仓库根目录执行：

```bash
bash tools/export_git_high1.sh /home/ubuntu/git-high1
```

缺 `.so` 时脚本会自动 cmake/make 重建（用 `ONNXRUNTIME_ROOT` 环境变量指定 ORT
预编译包位置，默认 `/opt/ort/onnxruntime-linux-x64-1.22.0`），然后把上面目录结构
完整复制到目标位置。
