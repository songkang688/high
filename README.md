# 证件照面部去高光（核心代码）

公安/证件照场景：减弱额头、鼻梁、脸颊、下巴等区域油光反光，尽量不改变五官与脸型。

本仓库为**核心源码 + 样例图 + 人脸关键点模型**，不含 Windows 便携包内置 Python（体积过大）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `highlight_removal/` | 核心算法：检测、划区域、去高光、质量检查 |
| `app.py` | Gradio 网页版 |
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

Python / ONNX 对比工作台（需 `pip install onnxruntime` 并按 `cpp/README.md` 编译 `libhigh_removal_ops.so`）：

```bash
python app_compare.py
```

浏览器打开：http://127.0.0.1:7861

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
「自定义算子封装外部推理运行时」方案：图中只有一个 `ai.high:HighlightRemoval` 自定义算子，
内核就是上面对拍验证过的 C++ 流水线，编译为 `libhigh_removal_ops.so`
（加载它是 ONNX Runtime 执行自定义算子的必需步骤，性质等同于安装 onnxruntime 本体）：

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("cpp/build/libhigh_removal_ops.so")   # 见 cpp/README.md 构建
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])

image = cv2.imread("data/1.png")                                     # uint8 [H, W, 3] BGR
result = sess.run(["result"], {"image": image})[0]                   # uint8 [H, W, 3] BGR
cv2.imwrite("out.png", result)
```

完整示例：`python tools/run_high_onnx.py -i data/1.png -o out.png`（Python）、
`cpp/build/high_onnx_session`（C++）。单 ONNX 会话输出与 C++ CLI **逐位相同**；
与原始 Python 流程的实测差异为 17 张平均 MAE 0.0028/255、PSNR 72.55 dB、掩码 IoU 0.9996
（非位级 100%，原因是推理引擎浮点尾差，见 `tools/PARITY_RESULTS.md`）。

## 说明

- 本仓库未包含体积约 50MB+ 的 OpenCV LBF 兜底模型 `lbfmodel.yaml`；正常使用 MediaPipe 模型即可。
- 不含 `dist/` 安装包与内置 Python。
