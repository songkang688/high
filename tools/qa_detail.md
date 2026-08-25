# 保护细节（onnx_detail / high_removal_detail.onnx）QA 报告

- QA 分支：`cursor/qa-onnx-detail-6544`（基于 `cursor/three-onnx-git-high1-6544`）
- 模型 slug：**claude-fable-5-thinking-xhigh**
- 复测脚本：`python tools/qa_detail.py`（与 `tools/three_onnx_audit.py` 相互独立的从零测量）
- 环境：Ubuntu 24.04 x64 纯 CPU；Python 3.12.3、opencv-contrib-python 4.14.0、
  onnxruntime 1.29.0、onnx 1.22.0、mediapipe 1.0.1；精确内核
  `cpp/build/libhigh_removal_pyops.so` 与 C++ 快速内核 `libhigh_removal_ops.so`
  由本次 QA 重新构建（g++ 13.3 + ONNX Runtime 1.22.0 头文件）。

## 总结论：**PASS**（12/12 项，含本次修复后复测）

未修改任何 strong / daily / default 文件；`configs/onnx_detail.yaml` 与
`models/high_removal_detail.onnx` 也**无需改动**（排序基线即通过，无需削弱 yaml
或重导出）。修复了一个真实的共享流水线 bug（只影响保护细节档，见下文 Bug 1）。

## 必测项逐条结果

| # | 必测项 | 结果 | 实测证据 |
|---|--------|------|----------|
| 1 | `high_removal_detail.onnx` 内嵌 `config_yaml` == `configs/onnx_detail.yaml` | ✅ PASS | 逐字节一致（onnx.load 读节点属性对比 read_bytes） |
| 2 | 精确内核 `session.run` 与 `process_image(onnx_detail.yaml)` 于 `data/1.png` `np.array_equal` | ✅ PASS | result 与 hard_mask 均逐位相同；另测连续两次 `session.run` 输出一致（确定性）✅ |
| 3 | `data/{1,15,16}.png` 上 detail < daily：掩码内 mean\|ΔL\| / 皮肤 L≥200 消减 / 整图 MAE | ✅ PASS 3/3 图 × 3 指标 | 见下表 |
| 4 | `simple_protect_mode: true`；`protect_expand_radius` 不小于 daily | ✅ PASS | 从**模型内嵌配置**（非仓库 yaml）读取：simple_protect_mode=true；protect_expand_radius detail=6 ≥ daily=5 |
| 5 | `data/1.png` 上与 daily、strong 的 session 输出均不逐位相同 | ✅ PASS | detail≠daily、detail≠strong（result+hard_mask 联合判定） |
| 6 | 共享 studio 问题记录（不改 app_studio.py） | ✅ 已记录 | 见「给集成者的共享问题记录」 |

### 强度指标（修复 Bug 1 后重测；各档用各自内嵌配置，指标在各自掩码/皮肤区内计算）

| 图 | 指标 | daily | detail | detail 更弱？ |
|----|------|-------|--------|---------------|
| 1.png | 掩码内 mean\|ΔL\| | 8.474 | **2.765** | ✅ |
| 1.png | 皮肤 L≥200 消减 | 9823 | **960** | ✅ |
| 1.png | 整图 MAE vs 原图 | 0.1812 | **0.0422** | ✅ |
| 15.png | 掩码内 mean\|ΔL\| | 5.969 | **1.919** | ✅ |
| 15.png | 皮肤 L≥200 消减 | 15616 | **1631** | ✅ |
| 15.png | 整图 MAE vs 原图 | 0.2027 | **0.0469** | ✅ |
| 16.png | 掩码内 mean\|ΔL\| | 11.436 | **3.260** | ✅ |
| 16.png | 皮肤 L≥200 消减 | 5446 | **435** | ✅ |
| 16.png | 整图 MAE vs 原图 | 0.2988 | **0.0579** | ✅ |

集成审计 `python tools/three_onnx_audit.py` 修复后同样整体 **PASS**
（强 > 日 > 细 两指标 3/3 图；三模型两两不逐位相同；daily == default 声明成立）。

## 发现并修复的 Bug

### Bug 1（已修复）：折中档 1/2 检测分辨率下，保护细节的形态学意图被静默反转

`configs/onnx_detail.yaml` 写明「不膨胀、反而腐蚀 1px，掩码收得更紧」
（`mask_dilate_radius: 0` + `mask_erode_radius: 1`）。但
`scaled_highlight_params`（`highlight_removal/utils.py`，C++ 侧
`cpp/src/utils.cpp` 同逻辑）在默认 `process_scale: compromise`（高光检测 1/2
分辨率）下：

- 对 `mask_dilate_radius` 施加**无条件下限 1** → 显式的 0 被强制抬成 1（反而膨胀）；
- 对 `mask_erode_radius` 直接 `round(1 × 0.5) = 0` → 请求的腐蚀被吞掉。

净效果与文档意图**完全相反**（实际执行「膨胀 1px + 不腐蚀」）。实测确认：修复前
`mask_erode_radius` 取 0/1/2 三个值在 `data/1.png` 上输出逐位相同（键值失效）。

**修复**（`highlight_removal/utils.py` + `cpp/src/utils.cpp` 同步）：形态学半径的
缩放下限只对显式启用（base > 0）的值生效——低分辨率下请求的膨胀/腐蚀/闭合不会
被四舍五入成 0，显式的 0 也不再被强制抬成 1。

**影响面验证**：

- default / daily / strong 在 3 张测试图上输出**逐位不变**（它们的 dilate>0、erode=0，
  下限行为等价）——满足「不改 strong/daily」约束；
- 保护细节档变化很小（结果图最大像素差 ≤4、差异像素数百级；三项强度指标变化
  ≤0.005，见 `tools/metrics_detail.md` 更新），方向为掩码边缘更收敛，与本档定位一致；
- 模型**无需重导出**：内嵌的是 yaml（未变），精确内核运行时调用仓库
  `process_image`，自动使用修复后代码；必测项 1–5 修复后全部复测通过；
- C++ 快速内核同步修改并重编译，`data/1.png` 上与 Python 侧 MAE 0.00097/255、
  掩码 IoU 0.9997（正常的亚像素级尾差范围）。

## 给集成者的共享问题记录（未在本分支修改）

1. **`tools/THREE_ONNX_AUDIT.md` 的 detail 行数字为修复前测量**：Bug 1 修复后
   detail 三项指标有 ≤0.005 的微调（如 1.png mean|ΔL| 2.761→2.765、L≥200 消减
   963→960；16.png 消减 426→435）。本 QA 已重跑 `tools/three_onnx_audit.py`
   确认整体仍 PASS；建议集成时重新生成该文档的数字。strong/daily/default 行不受影响。
2. **`cpp/build/windows/` 内已提交的预编译 DLL 基于修复前的 `cpp/src`**：
   `high_removal_ops.dll`（C++ 快速内核）在 Windows 上对 detail 档仍会执行旧的
   「膨胀 1 + 不腐蚀」行为（差异为数百像素、≤4/255 级别）。建议合并后重跑
   `.github/workflows/windows-dll.yml` 刷新 DLL。精确内核 DLL 语义不受影响
   （它调用宿主仓库的 Python 代码）。
3. **`highlight_removal/onnx_session.py` 的单槽会话缓存**：`get_session` 只缓存
   最近一个 (model, ops) 组合。app_studio 中来回切换强度预设时，每次切换都会
   重建 InferenceSession（重新加载 5.1 MiB 模型并注册算子库）。正确性无影响
   （键校验保证不会用错模型），纯性能问题；如需优化可改为按 key 的字典缓存。
4. **`onnx_session.availability()` 只检查 `models/high_removal.onnx`**：三个预设
   模型缺失时启动页仍显示「ONNX 引擎就绪」，运行到该预设才报错（错误会落到
   状态区，不崩溃）。建议 availability 顺带校验三个预设模型存在。
5. **调参提示（非 bug）**：detail 档在 3 张测试图上硬掩码面积恰好等于
   `highlight_max_area_ratio: 0.090` 的截断上限（12388/25504/19452 px，
   `highlight_detect.py` 超限后按 strength 分数取 top-N）。即该档掩码大小由面积
   上限主导，微调形态学/生长参数不会改变掩码面积，只会改变边缘像素的取舍；
   想改掩码大小需动 `highlight_max_area_ratio`。
6. **app_studio 界面小项（刻意未改）**：切换引擎仅切换面板可见性，旧结果图
   不清空（可能与新预设不对应）；`studio_config` 缓存的配置直接传给
   `process_image` 是安全的（`process_image` 内部 deepcopy，实测无跨请求污染）。

## 复现

```bash
pip install -r requirements.txt onnxruntime onnx
# 精确内核构建见 cpp/README.md（cmake + make high_removal_pyops）
python tools/qa_detail.py --json /tmp/qa_detail.json    # 退出码 0 = PASS
python tools/three_onnx_audit.py                        # 集成审计，亦 PASS
```
