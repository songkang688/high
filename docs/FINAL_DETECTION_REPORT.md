# 最终复测报告（FINAL DETECTION REPORT）

> 生成日期：2026-08-25 ｜ 分支：`cursor/studio-frontend-6544`（未合并 main，main 未改动）
> 性质：**检测 / 验证**报告。第一部分为全量回归复测（全部真实执行），第二部分为
> 「是否还有优化空间、是否存在观感更好的方案」的多角度实验（全部实测，含一个
> 3.4 GB 的 CVPR 2026 神经网络模型的真实下载与运行对比）。

---

## 1. 复测环境

本次复测在全新 VM 上从零搭建（未复用任何历史构建产物），四个 C++ 目标全部重新编译。

| 组件 | 版本 |
|------|------|
| 操作系统 / 编译器 | Ubuntu 24.04，g++ 13.3.0，CMake 3.28.3 |
| Python | 3.12.3 |
| numpy / opencv-contrib-python | 2.4.4 / **4.14.0**（requirements.txt 的 `<5` 约束） |
| onnxruntime (Python) / mediapipe / gradio | 1.29.0 / 1.0.1 / 6.26.0 |
| C++ 侧 OpenCV / yaml-cpp / ONNX Runtime | apt libopencv-dev **4.6.0** / 0.8.0 / 官方预编译包 **1.22.0** |
| 重新构建产物 | `libhigh_removal_pyops.so`、`libhigh_removal_ops.so`、`high_onnx`、`high_onnx_session`（一次 cmake/make 全部成功） |

与 `tools/PARITY_RESULTS.md` 记录的历史测试环境**完全一致**（同版本 Python/OpenCV/mediapipe/ORT/g++）。

---

## 2. 回归检测结果（全部真实执行）

### 2.1 精确内核：单 ONNX 会话 vs Python `process_image`（17 张 data/*.png）

`python tools/compare_python_onnx_session.py`，Python 侧配置 = `models/high_removal.onnx`
内嵌的 `configs/default.yaml`。

| 项目 | 结果 |
|------|------|
| 逐位相同（np.array_equal，含 hard_mask） | **17/17 ✅** |
| 平均 MAE / 最大像素差 | 0.0000 / 0 |
| 平均硬掩码 IoU | 1.0000 |

### 2.2 C++ 快速内核 vs Python（同 17 张）

| 项目 | 本次实测 | 历史记录 | 是否一致 |
|------|---------|---------|---------|
| 平均 MAE | 0.0028 | 0.0028 | ✅ |
| 平均 PSNR | 72.55 dB | 72.55 dB | ✅ |
| 平均硬掩码 IoU | 0.9996 | 0.9996 | ✅ |
| 全部图片最大像素差 | 10/255（3.png） | 10/255 | ✅ |
| 逐位相同 | 0/17（预期如此，浮点尾差） | 0/17 | ✅ |

逐图数字（MAE/max/IoU/PSNR 共 17 行）与 `tools/PARITY_RESULTS.md` 中的表**逐字节一致**：
两个对拍脚本（`compare_python_onnx_session.py`、`compare_python_cpp.py`）重新生成报告后
`git diff` 为空——因此本次**不需要更新** PARITY_RESULTS.md。

### 2.3 误差来源隔离 `python tools/parity_isolate.py`

| 项目 | 本次实测 |
|------|---------|
| 平均关键点误差（MediaPipe vs ORT landmarker，478 点） | max = 0.024 px / mean = 0.0077 px |
| 全链路 MAE（Python vs 单 ONNX C++ 内核） | 0.0028 |
| 仅关键点来源 MAE（同 OpenCV、只换关键点） | 0.0005 |
| 仅 CV 来源 MAE（关键点对齐后，pip 4.14 vs apt 4.6 + C++ 浮点路径） | **0.0026（主导）** |
| Python 自身确定性（同图两遍逐位相同） | 是 |
| CLI 常用模式 vs default.yaml 配置差异影响图片数 | 0/17 |

与历史记录完全一致。

### 2.4 工作室引擎函数（app_studio.py）

- `run_python_engine(data/1.png)` vs `run_onnx_engine`（精确内核）：result 与 hard_mask
  **均 np.array_equal = True**（耗时 Python 0.38 s / ONNX 0.33 s）；
- `HIGH_ONNX_KERNEL=cpp` 冒烟：session.run 正常返回 960×1280×3 结果 + 掩码（修改像素占全图 3.582%）；
- `import app_studio; app_studio.build_app()` 正常返回 `gr.Blocks`，不崩溃（gradio 6.26.0）。

### 2.5 C++ CLI

| 命令 | 结果 |
|------|------|
| `cpp/build/high_onnx --input data/1.png ... --config configs/default.yaml` | 成功（「处理成功」） |
| `cpp/build/high_onnx_session --model models/high_removal.onnx --ops libhigh_removal_ops.so` | 成功 |
| 两条路径输出 | **逐位相同**（np.array_equal = True） |

### 2.6 缺失 / 错误算子库的报错行为

`HIGH_OPS_LIB=/nonexistent/path.so` 时：

- `availability()` 返回 `(False, 中文提示)`，首行为
  「环境变量 HIGH_OPS_LIB=/nonexistent/path.so 指向的算子库不存在。」并附完整构建指引；
- `get_session()` 抛 `OnnxEngineUnavailable`（中文 message），**不崩溃**；
  `app_studio.run_studio` 捕获该异常写入状态区（页面不崩）。

### 2.7 结论

**7 项回归检测全部通过，无任何回归**；所有数字与分支上已记录的历史实测完全可复现。

---

## 3. 失败项

**无。** 唯一需要说明的非失败事项：本环境 `pip install opencv-contrib-python` 会默认装 5.0.0.93
（violates `<5`），须按 `requirements.txt` 安装才能复现历史环境（本次复测已按 requirements 安装 4.14.0.94）。

---

## 4. 优化空间与更好解的实验（全部实测）

实验脚本：`tools/detection_explore.py`、`tools/cv46_parity_experiment.py`（已入库）；
调试图与拼图：`runtime_outputs/detection_explore/`（已 gitignore，不入库）。
指标口径：|ΔL|/ΔE 为 OpenCV LAB uint8 口径下、按 default 配置 gated 硬掩码内的均值；
「皮肤 L≥200」为皮肤区亮斑像素计数（绝对口径的残余油光指标）；
「非高光皮肤 ΔE」衡量对无油光皮肤的附带伤害（越低越保真）。

### 4.A 强度预设对比（正常 / 强力 / 削弱 × 图 1 / 15 / 16）

**首要发现：「强力」预设与 default.yaml 的 `highlight_removal` 11 项参数逐项相等**，
三张图上「正常」与「强力」输出**逐位相同**——当前默认参数就是强力套；CLI「常用模式」
（= 强力 + 灵敏）与 default.yaml 的输出差异也已被 2.3 节证实为 0/17。

| 图 | 预设 | 掩码内 \|ΔL\| | ΔE | 掩码内均 L（前→后） | P95 L（前→后） | 皮肤 L≥200（前→后） | 警告 |
|----|------|-----------|----|---------------------|----------------|---------------------|------|
| 1  | 正常=强力 | 8.47 | 8.82 | 206.5→198.1 | 227→203 | 17368→7545（−57%） | 0 |
| 1  | 削弱 | 3.12 | 3.18 | 206.5→203.4 | 227→220 | 17368→14606（−16%） | 1 |
| 15 | 正常=强力 | 5.97 | 6.46 | 205.3→199.4 | 218→205 | 41563→25947（−38%） | 0 |
| 15 | 削弱 | 2.31 | 2.39 | 205.3→203.0 | 218→213 | 41563→37770（−9%） | 1 |
| 16 | 正常=强力 | 11.44 | 12.29 | 216.8→205.3 | 249→215 | 49396→43950（−11%） | 0 |
| 16 | 削弱 | 3.26 | 3.37 | 216.8→213.5 | 249→240 | 49396→47256（−4%） | 1 |

- 三套预设的**非高光皮肤 ΔE 均为 0.0000**（immutable 回写保证），五官/背景零改动；
- 「削弱」的 1 条警告为「当前修改区域过大，可能影响真实性，请降低处理强度」——它把
  `max_allowed_modify_area_ratio` 收紧到 0.09 而掩码约占脸 15%，属于阈值口径问题，非画质问题；
- **观感**（拼图 `runtime_outputs/detection_explore/16_montage_orig_正常_削弱.png`）：
  「正常/强力」显著压掉额头/鼻梁/脸颊大片油光且肤质自然；「削弱」残余油光明显可见。
- **回答任务问题「强力是否更好、会不会伤肤/伤五官」**：强力=默认，不存在更强一档；
  它没有伤害五官（保护区零改动、无质量审计警告），观感在三套里最好。

### 4.B C++ 内核逼近 Python 的剩余杠杆（同版本 OpenCV 实测验证）

2.3 节隔离显示残差主导来源是「仅 CV」（0.0026/0.0028）。本次**新增真实验证**
（`tools/cv46_parity_experiment.py`）：建独立 venv 装 **pip OpenCV 4.6.0**（与 C++ 侧 apt
4.6.0 同版本，numpy 1.26），注入与 C++ 内核同源的 ORT 关键点后跑原 Python 流水线，
与 C++ 内核输出直接对拍：

| 图 | 主环境「仅CV」残差（OpenCV 4.14 vs C++） | OpenCV 4.6 下 vs C++ | 变化 |
|----|------------------------------------------|----------------------|------|
| 1.png  | MAE 0.0026 | **MAE 0.0014**（max 5，0.236%） | 残差约减半 |
| 15.png | MAE 0.0061 | **MAE 0.0001**（max 2，0.015%） | 残差降 ~60× |
| 16.png | MAE 0.0055 | **MAE 0.0000，逐位相同 ✅** | 完全一致 |

结论（实测证明，不再是推断）：**匹配 OpenCV 版本确实是 C++ 内核逼近逐位一致的最后杠杆**；
剩余的亚 0.0014 残差来自 ORT 版本差（Python 1.29 vs C++ 1.22 的关键点网络浮点尾差）。
但需要逐位一致时精确内核已是 17/17，**不值得为此把 C++ 侧锁死在特定 OpenCV 版本**。

### 4.C 替代修复算法（同一检测掩码、同一保护逻辑，仅换修复核心；图 1 / 15）

| 图 | 方案 | 掩码内 \|ΔL\| | P95 L（前→后） | 皮肤 L≥200（前→后） | 非高光皮肤 ΔE |
|----|------|-----------|----------------|---------------------|----------------|
| 1  | **现有 TELEA+LAB 混合（基线）** | **8.47** | **227→203** | **17368→7545（−57%）** | 0.0000 |
| 1  | 导向滤波 guidedFilter(r=16, eps=15²) | 2.14 | 227→224 | 17368→14695（−15%） | 0.0000 |
| 1  | 频率分离（低频 inpaint 基底 + 纹理 100% 回加） | 4.69 | 227→216 | 17368→12092（−30%） | 0.0000 |
| 1  | 仅调参（suppress 0.94→0.80、chroma 0.38→0.55） | 8.29 | 227→203 | 17368→7966（−54%） | 0.0000 |
| 15 | **基线** | **5.97** | **218→205** | **41563→25947（−38%）** | 0.0000 |
| 15 | 导向滤波 | 1.86 | 218→214 | 41563→37465（−10%） | 0.0000 |
| 15 | 频率分离 | 3.79 | 218→211 | 41563→32977（−21%） | 0.0000 |
| 15 | 仅调参 | 5.85 | 218→205 | 41563→26596（−36%） | 0.0000 |

- **导向滤波明显更差**：边缘感知平滑会把大面积高光当作「结构」保留，压不掉油光块
  （观感上额头亮斑几乎原样，见拼图 `1_montage_orig_baseline_guided_freqsep_param.png`）；
- **频率分离居中**：纹理 100% 回加保住了皮肤颗粒感，但低频亮度压制弱于基线，残余油光可见；
- **仅调参无收益**：降低亮度压制 + 提高色度恢复 → 去油光变弱（−57%→−54%）、色度偏移增大
  （|Δa| 1.07→1.12、|Δb| 1.56→1.64），是轻微负优化；
- 备注：重跑同一套检测器测「残余高光面积」不具区分度（区域分位数自适应阈值恒圈出
  ~15% 人脸），因此以上采用绝对口径（均 L、P95、L≥200 计数）。

### 4.D 速度（data/1.png，960×1280，4 核 CPU，预热后 5 次取中位数）

| 路径 | 耗时 |
|------|------|
| raw `process_image`（Python 本体） | 281 ms |
| 单 ONNX session.run（精确内核） | 302 ms（≈ +7% 封送开销） |
| 单 ONNX session.run（C++ 内核） | 265 ms（≈ 比 Python 快 6%） |
| C++ CLI `high_onnx_session` 整进程 | ≈ 500 ms（含模型/库加载，纯推理与上行相当） |

**C++ 内核的速度优势可忽略**（瓶颈在 OpenCV 大核卷积/inpaint，两边同库同算法）；
C++ 内核的真正价值是**无 Python/MediaPipe 依赖的部署形态**，不是速度。

### 4.E 架构结论 + 神经网络方案实测（真实下载并运行）

架构现状（复测确认）：

1. **精确内核** = 同一 `models/high_removal.onnx` + `libhigh_removal_pyops.so`，Compute 内
   调用原始 Python 流水线本体 → 与 `process_image` **100% 逐位一致**，但宿主必须是 Python 进程；
2. **C++ 内核** = 同一 ONNX + `libhigh_removal_ops.so`，无 Python/MediaPipe 依赖，
   MAE ≈ 0.0028（4.B 已实测证明残差主导是 OpenCV 版本差异）；
3. **纯标准算子的单一 ONNX 图仍不可行**：TELEA inpaint（快速行进法的数据依赖迭代）与
   连通域分析（不定迭代次数的标号传播）无法用标准 ONNX 算子表达，结论不变。

**神经网络方案实测**（回答「深度学习去高光会不会观感更好」）：下载并在 CPU 上运行了
CVPR 2026 Oral 的 **UnReflectAnything**（当前公开 SOTA；权重 3.4 GB，DINOv3 ViT 编码器 +
token inpainting，448×448 输入。torch 2.13+cpu；官方包 v1.0.2 需将 transformers 降到 4.57
才能加载权重——5.x 的 DINOv3 键名带 `model.` 前缀不匹配）：

| 指标（1.png，同一 gated 掩码口径） | 现有流水线 | UnReflectAnything | UnReflectAnything (composite) |
|-------------------------------------|-----------|-------------------|-------------------------------|
| 掩码内 P95 L（前→后） | **227→203** | 227→214 | 227→216 |
| 皮肤 L≥200（前→后） | **−57%** | −31% | −14% |
| 非高光皮肤 ΔE（应≈0） | **0.0000** | 3.98 | 1.15 |
| 背景 ΔE（应≈0） | **0.0000** | 2.46 | 0.96 |
| 单张耗时（CPU） | **0.28 s** | ≈ 20 s | ≈ 20 s |
| 部署体积 | ~35 MB（模型+库） | 3.4 GB | 3.4 GB |

**观感**（拼图 `1_neural_montage_*.png`、`1_neuralcomp_montage_*.png`）：神经网络输出
经 448×448 重建后整脸变糊、额头/鼻尖出现**红色斑块伪影**，五官纹理受损；composite
模式保住了掩码外区域但斑块伪影仍在。**在证件照场景下全面差于现有经典流水线**
（去油光更弱、保真更差、慢 70 倍、体积大 100 倍）。该模型面向自然/手术图像的通用
高光，不是为人脸证件照的「保结构去油光」设计的。

---

## 5. 建议

1. **不合并 main**（遵守约束，本次也未合并）；当前分支交付形态已完备：
   需要逐位一致用精确内核，需要脱离 Python 部署用 C++ 内核，两者共用一个 ONNX 文件。
2. **保持现有默认算法与参数（= 强力套）**：在三套预设与三种替代算法、以及 2026 年
   SOTA 神经网络模型的实测对比中，现有 TELEA+LAB 混合方案的去油光力度、保真度
  （掩码外零改动）与观感全面最优。**不建议**换导向滤波（实测几乎压不掉油光）、
   频率分离（力度不足）或该神经网络模型（伪影 + 全图失真 + 3.4 GB）。
3. **配置卫生（可选小改）**：`removal_intensity_presets.yaml` 的「强力」与 default.yaml
   完全重复——要么让「强力」真的更强（如 suppress 0.94→0.97、blend 0.97→1.0），要么在
   注释里说明「默认即强力」；「削弱」的 `max_allowed_modify_area_ratio: 0.09` 必然触发
  「修改区域过大」警告，建议同步放宽或接受该提示。
4. **若未来还要更好观感，值得动的方向（按性价比）**：
   a. 针对 16.png 这类**极端大片油光**（P95 249），在极端核心的二次 inpaint 后增加
      局部色度一致性约束（当前 |Δa|2.2/|Δb|3.2 是三张里最高的），减少极亮区修复后的轻微偏色；
   b. 训练/微调一个**小型人脸专用**去油光网络（以本流水线输出为教师、或成对棚拍数据），
      而不是套用通用高光移除大模型——4.E 实测证明通用大模型在此场景不可用；
   c. C++ 内核如需进一步逼近 Python：升级 apt 侧 OpenCV 至 4.14（或 pip 侧钉住 4.6）
      可把 MAE 从 0.0028 压到 ≤0.0014 甚至逐位一致（4.B 实测），但收益有限，不作为必做项。
5. **无需更新** `tools/PARITY_RESULTS.md`：两个生成脚本重跑后文件逐字节不变。

---

### 附：本次复测执行的命令清单

```bash
# 构建
cd cpp && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 .. && make -j4
# 回归（PART 1）
python tools/compare_python_onnx_session.py     # 精确 17/17 逐位一致 + C++ MAE 0.0028
python tools/compare_python_cpp.py              # 旧 CLI 全量对拍（数字不变）
python tools/parity_isolate.py                  # 关键点 0.0005 vs CV 0.0026 拆解（数字不变）
LD_LIBRARY_PATH=/opt/ort/onnxruntime-linux-x64-1.22.0/lib ./cpp/build/high_onnx -i data/1.png -o out.png -c configs/default.yaml
LD_LIBRARY_PATH=/opt/ort/onnxruntime-linux-x64-1.22.0/lib ./cpp/build/high_onnx_session --model models/high_removal.onnx --ops cpp/build/libhigh_removal_ops.so --input data/1.png --output out2.png
# 探索（PART 2）
python tools/detection_explore.py               # 预设/替代算法/速度
/tmp/cv46venv/bin/python tools/cv46_parity_experiment.py   # OpenCV 4.6 同版本对拍
```
