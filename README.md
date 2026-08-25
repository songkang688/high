# 证件照面部去高光（核心代码）

公安/证件照场景：减弱额头、鼻梁、脸颊、下巴等区域油光反光，尽量不改变五官与脸型。

本仓库为**核心源码 + 样例图 + 人脸关键点模型**，不含 Windows 便携包内置 Python（体积过大）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `highlight_removal/` | 核心算法：检测、划区域、去高光、质量检查 |
| `app_studio.py` | **推荐 UI**：Python / ONNX 引擎切换与并排对比工作室 |
| `app.py` | Gradio 网页版（全参数调试台） |
| `app_simple.py` | PyQt5 桌面简化版 |
| `cli_process.py` + `run_facehi_terminal.bat` | 命令行入口 |
| `configs/` | 默认参数与预设 |
| `models/face_landmarker.task` | MediaPipe 人脸关键点模型 |
| `models/*.onnx` | 由 `.task` 导出的 ONNX 人脸模型（供 C++ 版使用） |
| `data/` | 样例证件照 |
| `cpp/` | C++ 版推理链路（OpenCV + ONNX Runtime，见 [`cpp/README.md`](cpp/README.md)） |
| `docs/ONNX_CPP_PLAN.md` | ONNX/C++ 落地方案与「能否整体转 ONNX」的结论 |
| `CLI使用说明.txt` | 命令行用法 |

## 环境与启动

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

浏览器打开：http://127.0.0.1:7860

### 工作室页（推荐 UI）：Python / ONNX 切换与对比

```bash
python app_studio.py
```

浏览器打开：http://127.0.0.1:7861。三种模式：**Python**（原始链路）、**ONNX**
（`models/high_removal.onnx` + 自定义算子库）、**对比**（同图双引擎并排 +
差异热力图 + MAE / PSNR / 掩码 IoU / 耗时）。ONNX 侧默认使用**精确内核**
`libhigh_removal_pyops.so`（`session.run` 输出与 Python 引擎**逐位相同**，对比模式
MAE 恒为 0）；设 `HIGH_ONNX_KERNEL=cpp` 可切换到 C++ 快速内核 `libhigh_removal_ops.so`
（存在亚像素级浮点尾差），`HIGH_OPS_LIB` 可直接指定 .so 路径。两个库由同一次
[`cpp/README.md`](cpp/README.md) 构建产出；缺失时页面会给出构建提示，Python 模式不受影响。
监听地址 / 端口可用 `HIGH_STUDIO_HOST`（默认 127.0.0.1）、`HIGH_STUDIO_PORT`（默认 7861）覆盖。

桌面版：

```bash
python app_simple.py
```

命令行：

```bash
python cli_process.py -i data/1.png
```

## 处理流程（简述）

1. 人脸关键点定位（MediaPipe）
2. 划分高光候选区与保护区
3. 高光检测
4. 局部去高光修复
5. 真实性检查

## C++ / ONNX 版

推理链路已完整移植为 C++（OpenCV + ONNX Runtime，纯 CPU），与 Python 输出逐像素高度一致
（17 张样例平均 PSNR 72.6 dB、高光掩码 IoU 0.9996）。构建与运行见 [`cpp/README.md`](cpp/README.md)，
方案与「Python 流程能否整体转成一个 ONNX 模型」的结论见 [`docs/ONNX_CPP_PLAN.md`](docs/ONNX_CPP_PLAN.md)，
实测对拍数据见 [`tools/PARITY_RESULTS.md`](tools/PARITY_RESULTS.md)。

## 单 ONNX 模型调用（推荐的最终交付形态）

整条流水线已封装为**一个** ONNX 文件 `models/high_removal.onnx`（内嵌两个人脸网络权重与
`configs/default.yaml` 全部配置）。因为 TELEA inpaint、连通域分析、动态分位数阈值等环节
**无法用标准 ONNX 算子表达**（详见 `docs/ONNX_CPP_PLAN.md` 第一节），这里采用微软官方的
「自定义算子封装外部推理运行时」方案：图中只有一个 `ai.high:HighlightRemoval` 自定义算子节点。
同一个 ONNX 文件配两个**可互换**的内核库（注册哪个就用哪个，模型文件不变）：

| 内核库 | 与 Python 原始流程的一致性（17 张实测） | 适用场景 |
|--------|----------------------------------------|---------|
| `libhigh_removal_pyops.so`（**精确内核，默认**） | **17/17 逐位相同**（np.array_equal，MAE 0、最大差 0、掩码 IoU 1.0）——Compute 内直接调用原始 Python 流水线本体 | Python 宿主（`onnxruntime` + 本仓库 + mediapipe），要求 100% 等效 |
| `libhigh_removal_ops.so`（C++ 快速内核） | 平均 MAE 0.0028/255、PSNR 72.55 dB、掩码 IoU 0.9996、最大单像素差 10/255 | 纯 C++ 部署 / 无 Python 环境，可接受亚像素级浮点尾差 |

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
# 精确内核（与 Python process_image 逐位相同；见 cpp/README.md 构建）：
so.register_custom_ops_library("cpp/build/libhigh_removal_pyops.so")
# 或 C++ 快速内核（无 Python/MediaPipe 依赖）：
# so.register_custom_ops_library("cpp/build/libhigh_removal_ops.so")
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])

image = cv2.imread("data/1.png")                                     # uint8 [H, W, 3] BGR
result = sess.run(["result"], {"image": image})[0]                   # uint8 [H, W, 3] BGR
cv2.imwrite("out.png", result)
```

完整示例：`python tools/run_high_onnx.py -i data/1.png -o out.png`（Python，`--kernel exact/cpp`）、
`cpp/build/high_onnx_session`（C++，只能用 C++ 内核）。精确内核要求宿主为 Python 进程且本仓库可导入
（默认按 .so 位置自动定位仓库，也可用 `HIGH_PY_KERNEL_PATH` 指定）；C++ 内核输出与 C++ CLI **逐位相同**。
全部实测数字与误差来源拆解见 `tools/PARITY_RESULTS.md`。

## 说明

- 本仓库未包含体积约 50MB+ 的 OpenCV LBF 兜底模型 `lbfmodel.yaml`；正常使用 MediaPipe 模型即可。
- 不含 `dist/` 安装包与内置 Python。
