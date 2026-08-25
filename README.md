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
| `data/` | 样例证件照 |
| `CLI使用说明.txt` | 命令行用法 |

## 环境与启动

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

浏览器打开：http://127.0.0.1:7860

桌面版：

```bash
python app_simple.py
```

命令行（需先装好依赖）：

```bash
python cli_process.py -i data/1.png
```

## 处理流程（简述）

1. 人脸关键点定位（MediaPipe，失败可用本地兜底）
2. 划分高光候选区与保护区
3. 高光检测（亮度/色彩/区域自适应）
4. 局部去高光修复
5. 真实性检查

## 说明

- 默认工作目录中的样例图相对路径为 `../data` 时，请把本仓库的 `data/` 放在与程序目录同级，或改会话配置中的路径。
- 本仓库未包含体积约 50MB+ 的 OpenCV LBF 兜底模型 `lbfmodel.yaml`；正常使用 MediaPipe 模型即可。
