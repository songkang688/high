# 三 ONNX 变体独立审计（强力 / 日常 / 保护细节）：**PASS**

独立复测：`python tools/three_onnx_audit.py`（2026-08-25，与三个变体分支各自的实测脚本
无关的从零测量）。环境：Python 3.12.3、onnxruntime 1.29.0、opencv-contrib-python 4.14.0、
mediapipe 1.0.1；精确内核 `cpp/build/libhigh_removal_pyops.so` 由本次审计重新构建
（g++ 13.3 + ONNX Runtime 1.22.0 头文件，Ubuntu 24.04）。

> **本轮（第二次 QA 合并）更新**：合入 `cursor/qa-onnx-strong-6544`、
> `cursor/qa-onnx-daily-6544`、`cursor/qa-onnx-detail-6544` 三个 QA 分支后全量重测。
> detail QA 分支带来一个真实修复（`highlight_removal/utils.py` + `cpp/src/utils.cpp`）：
> `scaled_highlight_params` 原来对 `morph_close_radius`/`mask_dilate_radius` 施加
> **无条件下限 1**、对 `mask_erode_radius` 不设下限，导致保护细节档在折中档 1/2 检测
> 分辨率下显式的 `mask_dilate_radius: 0` 被强制抬成 1、`mask_erode_radius: 1` 被四舍五入
> 成 0（形态学意图被反转）。修复后下限只对显式启用（base > 0）的形态学半径生效并覆盖
> erode。本次实测确认（缩放后参数打印）：detail 在 scale=0.5 下 dilate `0→0`、erode `1→1`；
> daily/strong/default 的缩放结果与修复前逐项相同（dilate 2→1 / 3→2、erode 0→0、
> close 4→2 / 5→2）。**输出层面**：与合并前 `origin/cursor/three-onnx-git-high1-6544`
> 的 `process_image` 输出在 1/15/16.png 上逐哈希对比，daily/strong/default 全部
> **逐位不变**，仅 detail 按预期微调（下表已按修复后代码重测，变化 ≤0.005 量级）。

## 审计对象

| 预设 | 模型 | 内嵌配置 | 体积 |
|------|------|---------|------|
| 强力 | `models/high_removal_strong.onnx` | `configs/onnx_strong.yaml` | 5.10 MiB |
| 日常 | `models/high_removal_daily.onnx` | `configs/onnx_daily.yaml` | 5.10 MiB |
| 保护细节 | `models/high_removal_detail.onnx` | `configs/onnx_detail.yaml` | 5.10 MiB |

## 判定结果（全部通过）

| # | 判据 | 结果 |
|---|------|------|
| 1 | 三个 ONNX 存在（各 5.10 MiB）、含 `config_yaml` 属性且与对应 `configs/onnx_*.yaml` **逐字节一致** | ✅ 3/3 |
| 2 | `data/1.png` 上三模型 session.run（精确内核）输出两两**不**逐位相同（强力≠日常、日常≠细节、强力≠细节） | ✅ 3/3 对均不同 |
| 3 | 去油光强度排序 强力 > 日常 > 保护细节（两个指标独立判定，要求 ≥2/3 图片） | ✅ mean\|ΔL\| 3/3 图、L≥200 消减 3/3 图（双指标全满足） |
| 4 | `data/1.png` 上三模型 session.run 与 `process_image(对应 yaml)` `np.array_equal`（result 与 hard_mask） | ✅ 3/3 |
| 5 | 面部保护开启（三份内嵌配置 `simple_protect_mode: true`）；全程无崩溃 | ✅ |
| 6 | **utils 修复后无回归**：daily/strong/default 输出与合并前分支逐位相同（3 图 sha256）；detail 缩放参数 dilate 保持 0、erode 保持 ≥1 | ✅ |

变体声明附加验证（修复后重测）：日常版与 `default.yaml` 流水线在 3 张图上逐位相同 ✅；
`high_removal_daily.onnx` 与现有 `high_removal.onnx`（同一精确内核）在 1.png 上逐位相同 ✅。

预设接线复测：`app_studio.py` 预设 Radio（强力/日常/保护细节，默认日常）UI 构建正常，
三预设下 Python 引擎与 ONNX 引擎输出逐位相同、三预设两两不同 ✅；
`tools/run_high_onnx.py --preset strong|daily|detail` 全部成功，输出与 app_studio
ONNX 引擎逐位一致 ✅。C++ 快速内核（`libhigh_removal_ops.so`，含同一修复）在 detail
模型上冒烟通过：硬掩码与精确内核 IoU 0.9997（亚像素级浮点尾差，符合预期）✅。

## 独立实测数据（process_image，各预设用各自内嵌配置）

L = OpenCV LAB 的 L 通道（0~255）；掩码内 mean|ΔL| = 各自高光硬掩码内处理前后 L 的平均
绝对变化；皮肤 L≥200 = 流水线肤色掩码内亮斑像素数（前→后，「消减」越大去油光越强）；
MAE = 整图三通道 vs 原图。

### data/1.png

| 预设 | 硬掩码 px | 掩码内 mean\|ΔL\| | 皮肤 L≥200 前→后（消减） | MAE vs 原图 |
|------|-----------|-------------------|--------------------------|-------------|
| 强力 | 30288 | **10.047** | 17368→385（**16983**） | 0.2954 |
| 日常 | 21336 | 8.474 | 17368→7545（9823） | 0.1812 |
| 保护细节 | 12388 | 2.765 | 17368→16408（960） | 0.0422 |
| （default 参考） | 21336 | 8.474 | 17368→7545（9823） | 0.1812 |

### data/15.png

| 预设 | 硬掩码 px | 掩码内 mean\|ΔL\| | 皮肤 L≥200 前→后（消减） | MAE vs 原图 |
|------|-----------|-------------------|--------------------------|-------------|
| 强力 | 62344 | **6.767** | 41563→12699（**28864**） | 0.3175 |
| 日常 | 43924 | 5.969 | 41563→25947（15616） | 0.2027 |
| 保护细节 | 25504 | 1.919 | 41560→39929（1631） | 0.0469 |
| （default 参考） | 43924 | 5.969 | 41563→25947（15616） | 0.2027 |

### data/16.png

| 预设 | 硬掩码 px | 掩码内 mean\|ΔL\| | 皮肤 L≥200 前→后（消减） | MAE vs 原图 |
|------|-----------|-------------------|--------------------------|-------------|
| 强力 | 47552 | **11.659** | 49396→34100（**15296**） | 0.4247 |
| 日常 | 33500 | 11.436 | 49396→43950（5446） | 0.2988 |
| 保护细节 | 19452 | 3.260 | 49391→48956（435） | 0.0579 |
| （default 参考） | 33500 | 11.436 | 49396→43950（5446） | 0.2988 |

三张图上两项排序指标均严格满足 强力 > 日常 > 保护细节，且日常与 default 参考行
数值完全一致（逐位相同的必然结果）。强力/日常/default 三行与修复前审计数值完全一致；
保护细节的硬掩码 px 不变（该档掩码在这三张图上被 `highlight_max_area_ratio: 0.090`
截断到面积上限，形状微调但总数不变），mean|ΔL|、L≥200 消减、MAE 有 ≤0.005 量级微调。

## 复现

```bash
# 依赖：pip install -r requirements.txt onnxruntime onnx；
# 精确内核构建见 cpp/README.md（cmake + make high_removal_pyops）。
python tools/three_onnx_audit.py --json /tmp/three_onnx_audit.json
```

脚本退出码 0 = PASS；任何一条判据不满足则打印 `[FAIL]` 明细并以退出码 1 结束。
detail 档修复细节与复测脚本见 `tools/qa_detail.md` / `tools/qa_detail.py`；
强力、日常两档的 QA 报告见 `tools/qa_strong.md` / `tools/qa_daily.md`。
