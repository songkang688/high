# git-high2 三引擎平均用时对照（data/1.png…17.png 全量）

复现：`python git-high2/bench/bench_runtime.py --all`
（原始明细：`bench_raw.csv`；机器可读汇总：`bench_summary.json`）

## 环境

| 项 | 值 |
| --- | --- |
| CPU | 4 逻辑核（Linux x86-64 云环境） |
| Python | 3.12.3 |
| numpy / OpenCV / onnxruntime | 2.4.4 / 4.14.0 / 1.29.0 |
| mediapipe | 1.0.1 |
| 自定义算子库 | `git-high2/lib/libfacehi_custom_ops.so`（已提交产物） |

## 引擎定义与协议

| 引擎 | 实现 |
| --- | --- |
| `python_orig` | `cli_process.load_cli_config("常用模式")` + `highlight_removal.pipeline.process_image`（MediaPipe FaceLandmarker） |
| `python_studio` | **与 `python_orig` 为同一条链路，不单列**。git-high2 前端「Python 版」页签（`app_git_high2.py` 的 `_run_python`）就是 `load_cli_config(mode)` + `process_image(bgr, cfg)`，无第三条实现 |
| `onnx_python` | `git-high2/facehi_onnx.py` 的 `FacehiOnnx`：`onnxruntime` + `register_custom_ops_library` + `models/facehi.onnx`，**同一进程复用 session** |

公平性要点：

1. 每个引擎独立子进程，先用 `data/1.png` warmup 1 张（含模型/会话加载，计冷启动），
   再对 17 张逐张计时；只计处理调用，图片解码在计时外。
2. 线程与常用模式一致：`python_orig` 走 `bootstrap_before_numpy()`（CPU 狂暴 =
   全部逻辑核）+ `load_cli_config` 内 `cv2.setNumThreads(4)`；`onnx_python` 用
   `FacehiOnnx` 文档默认 `intra_op_threads=0`（ORT 自选，物理核数）。
3. 可视化关闭（`load_cli_config` 固定 `enable_visualization=False`，CLI 常用即此）。
4. 测速时机器空闲（交叉编译等重活在测速完成后才启动）。

## 汇总（单位：秒）

| 引擎 | 冷启动（加载+首张） | 热平均 | 热中位数 | p95 | 最小 | 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| python_orig | 1.359（加载 0.872 + 首张 0.486） | **0.2854** | 0.2787 | 0.5087 | 0.0794 | 0.5087 |
| onnx_python | 0.535（加载 0.271 + 首张 0.265） | **0.2234** | 0.2259 | 0.4110 | 0.0652 | 0.4110 |

**结论：ONNX 热平均 0.2234s vs 原 Python 热平均 0.2854s，ONNX 快约 21.7%
（0.2234 / 0.2854 ≈ 0.783），冷启动也快 2.5 倍。「ONNX 不慢于原 Python」成立，
无需再查线程 / HWC 拷贝 / session 重建（session 已全程复用）。**

## 每张明细（单位：秒）

| 图片 | 分辨率 | python_orig | onnx_python | ONNX/Python |
| --- | --- | --- | --- | --- |
| 1.png | 1280×960 | 0.3442 | 0.2331 | 0.68 |
| 2.png | 1280×960 | 0.3416 | 0.2647 | 0.77 |
| 3.png | 648×486 | 0.0812 | 0.0686 | 0.85 |
| 4.png | 1280×960 | 0.2926 | 0.2276 | 0.78 |
| 5.png | 1280×960 | 0.2400 | 0.1839 | 0.77 |
| 6.png | 648×486 | 0.0794 | 0.0652 | 0.82 |
| 7.png | 1280×960 | 0.3289 | 0.2706 | 0.82 |
| 8.png | 1280×960 | 0.2620 | 0.1856 | 0.71 |
| 9.png | 1280×960 | 0.2787 | 0.2164 | 0.78 |
| 10.png | 1280×960 | 0.2658 | 0.2119 | 0.80 |
| 11.png | 1280×960 | 0.2518 | 0.1960 | 0.78 |
| 12.png | 1280×960 | 0.2958 | 0.2419 | 0.82 |
| 13.png | 1280×960 | 0.2780 | 0.2259 | 0.81 |
| 14.png | 1448×1086 | 0.3302 | 0.2398 | 0.73 |
| 15.png | 1324×1253 | 0.5087 | 0.4110 | 0.81 |
| 16.png | 1448×1086 | 0.4078 | 0.3384 | 0.83 |
| 17.png | 1280×960 | 0.2653 | 0.2164 | 0.82 |

17/17 张 ONNX 均不慢于 Python（逐张比值 0.68–0.85）。

## 正确性抽检（data/1.png，ONNX vs Python 完整链路）

| 指标 | 值 | 与既有报告对照 |
| --- | --- | --- |
| MAE | 0.00318 | README「验证记录」：0.0032 ✓ |
| 最大像素差 | 7 / 255 | 同 7/255 ✓ |
| 差异像素占比 | 0.504% | 同 0.504% ✓ |
| 硬掩码 IoU | 0.9966 | 同 0.9966 ✓ |

差异量级与既有报告一致，来源为两套人脸推理引擎（MediaPipe vs 内嵌 ONNX
检测/关键点）的亚像素关键点噪声；注入同一关键点时两边位级一致（见
`onnx/tests/onnx_report.md`）。
