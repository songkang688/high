# git-high1 · 证件照去高光便携包（三档强度 ONNX + 前端 + 调用脚本）

本目录是 `high` 仓库的**自包含可运行子集**：四个单文件模型（每个都内嵌两个人脸网络
与各自的 yaml 配置）+ 工作室前端 `app_studio.py` + 命令行调用脚本
`tools/run_high_onnx.py`，以及运行所需的算子库与示例图。

三档去油光强度预设（独立实测对比见仓库 `tools/THREE_ONNX_AUDIT.md`）：

| 预设 | 模型 | 内嵌配置 | 效果 |
|------|------|---------|------|
| 强力 | `models/high_removal_strong.onnx` | `configs/onnx_strong.yaml` | 去油光最强（皮肤 L≥200 亮斑消减最多） |
| 日常（默认） | `models/high_removal_daily.onnx` | `configs/onnx_daily.yaml` | = 当前 `default.yaml` 平衡效果，输出与 `high_removal.onnx` 逐位相同 |
| 保护细节 | `models/high_removal_detail.onnx` | `configs/onnx_detail.yaml` | 去油光最弱、保留更多皮肤纹理（掩码内 \|ΔL\| 最低） |

`models/high_removal.onnx`（内嵌 `configs/default.yaml`）继续保留，输出与日常档逐位相同。

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
  app_studio.py            # 前端：强力/日常/保护细节 预设 × Python / ONNX / 对比 三种引擎
  tools/run_high_onnx.py   # 命令行调用入口（session.run 示例，零仓库依赖，--preset 选强度）
  models/high_removal.onnx         # 单文件模型（内嵌人脸网络 + default.yaml，5.1 MB）
  models/high_removal_strong.onnx  # 强力档（内嵌 onnx_strong.yaml，5.1 MB）
  models/high_removal_daily.onnx   # 日常档（内嵌 onnx_daily.yaml，5.1 MB；= default 效果）
  models/high_removal_detail.onnx  # 保护细节档（内嵌 onnx_detail.yaml，5.1 MB）
  models/face_landmarker.task      # MediaPipe 人脸模型（精确内核 / Python 引擎需要）
  configs/*.yaml           # default.yaml + 三档预设 onnx_{strong,daily,detail}.yaml 等
  highlight_removal/       # Python 包（精确内核经它调用原始流水线 process_image）
  cpp/build/libhigh_removal_pyops.so  # 精确内核（效果最好：与 Python 逐位一致）
  cpp/build/libhigh_removal_ops.so    # C++ 快速内核（无 Python/MediaPipe 依赖）
  cpp/build/windows/high_removal_pyops.dll  # 精确内核 Windows 版（存在时；来自 windows-dll workflow）
  cpp/build/windows/high_removal_ops.dll    # C++ 快速内核 Windows 版（OpenCV/yaml-cpp 已静态链接）
  data/                    # 17 张示例图（前端下拉框直接可选）
```

Windows 上无需改代码：`tools/run_high_onnx.py` 与 `highlight_removal/onnx_session.py`
会按平台自动选 `.so`（Linux）或 `.dll`（Windows，搜索 `cpp/build/`、`cpp/build/Release/`、
`cpp/build/windows/`）。

## 依赖

```bash
pip install onnxruntime opencv-contrib-python mediapipe gradio numpy PyYAML Pillow scikit-image
```

只用 **C++ 快速内核**跑命令行时最小依赖仅为 `onnxruntime opencv-python numpy`；
精确内核 / Python 引擎 / 前端 需要完整一行（mediapipe、gradio 等）。

版本注意：精确内核任何 pip onnxruntime ≥ 1.22 均可；**C++ 快速内核在 Python 宿主下
需要 pip onnxruntime ≥ 1.23**（1.22.x 的 Python 绑定与内核嵌套建会话存在 LoggingManager
冲突，详见仓库 `cpp/README.md`）。

## 启动前端

```bash
cd /home/ubuntu/git-high1
python app_studio.py                    # http://127.0.0.1:7861
HIGH_STUDIO_HOST=0.0.0.0 HIGH_STUDIO_PORT=8080 python app_studio.py   # 改地址/端口
HIGH_ONNX_KERNEL=cpp python app_studio.py                             # ONNX 侧改用 C++ 快速内核
```

页面提供三档强度预设（**强力 / 日常 / 保护细节**，默认日常；Python 引擎自动加载对应
yaml，对比模式两侧参数一致）× 三种引擎：**Python**（MediaPipe + OpenCV 原始链路）、
**ONNX**（所选预设的单文件模型 + 自定义算子库）、**对比**（同图并排跑两个引擎，
显示差异热力图与 MAE/PSNR/IoU 指标）。

## 命令行调用

```bash
cd /home/ubuntu/git-high1
python tools/run_high_onnx.py -i data/1.png -o out.png                # 精确内核（默认配置）
python tools/run_high_onnx.py --preset strong -i data/1.png -o out.png  # 强力档
python tools/run_high_onnx.py --preset daily -i data/1.png -o out.png   # 日常档（= 默认效果）
python tools/run_high_onnx.py --preset detail -i data/1.png -o out.png  # 保护细节档
python tools/run_high_onnx.py --kernel cpp -i data/1.png -o out.png   # C++ 快速内核
python tools/run_high_onnx.py -i data/1.png -o out.png --mask mask.png  # 另存高光硬掩码
```

## session.run 直接调用（核心就这几行）

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("cpp/build/libhigh_removal_pyops.so")  # 或 libhigh_removal_ops.so
# Windows 用 DLL：
# so.register_custom_ops_library("cpp/build/windows/high_removal_pyops.dll")  # 或 high_removal_ops.dll
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
result, hard_mask = sess.run(["result", "hard_mask"], {"image": cv2.imread("data/1.png")})
```

换强度档只需换模型文件名：`models/high_removal_strong.onnx` /
`models/high_removal_daily.onnx` / `models/high_removal_detail.onnx`（配置已各自内嵌）。

接口：输入 `image` uint8 `[H, W, 3]`（BGR，`cv2.imread` 原样，H/W 动态）；
输出 `result` uint8 `[H, W, 3]`（去高光结果）、`hard_mask` uint8 `[H, W]`（高光硬掩码）。

## 精确内核 vs C++ 快速内核

同一个 `high_removal.onnx` 支持两个可互换的内核库——注册哪个 `.so` 就用哪个内核，
模型文件不需要任何改动：

| 内核 | 库文件（Linux / Windows） | 效果 | 依赖 |
|------|--------|------|------|
| **精确内核（默认，效果最好）** | `libhigh_removal_pyops.so` / `high_removal_pyops.dll` | 与 Python 流水线 `process_image` **逐位相同**（17/17 张 `np.array_equal`，MAE 恒为 0） | 宿主必须是 Python 进程（≥ 3.10）；需要本目录的 `highlight_removal/` 包、`models/face_landmarker.task` 与 mediapipe |
| C++ 快速内核 | `libhigh_removal_ops.so` / `high_removal_ops.dll` | 完整 C++ 移植，存在亚像素级浮点尾差（17 张实测平均 MAE ≈ 0.003/255，PSNR ≈ 72 dB，掩码 IoU ≈ 0.9996） | 无 Python/MediaPipe 依赖，可被纯 C++ ORT 宿主加载；Linux 运行期需系统 OpenCV 与 yaml-cpp 运行库，Windows DLL 已静态链接二者（仅需 VC++ 运行库） |

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
完整复制到目标位置；仓库 `cpp/build/windows/` 下若有 Windows DLL 也会一并打包。

Windows DLL 的构建方式（GitHub Actions `windows-dll` workflow 或本机 VS 2022 + vcpkg）
见仓库 `cpp/README.md` 的「Windows 构建」一节。
