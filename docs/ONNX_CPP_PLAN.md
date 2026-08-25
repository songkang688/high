# 去高光流程 ONNX / C++ 落地方案

> 目标：让 C++ 侧调用后得到与当前 Python 流程一致的去高光结果。
> 本文档先回答「Python 流程能否整体转成一个 ONNX 模型」，再给出选定的落地架构与实现计划。

---

## 一、结论（TL;DR）

**问：现在的 Python 去高光流程能不能转成一个 ONNX 模型，C++ 只调一次就得到 100% 相同的结果？**

**答：不能。** 本仓库不是一个神经网络，而是「神经网络关键点 + 大量传统 OpenCV 图像算法」的混合流水线。其中以下环节**无法**用标准 ONNX 算子（官方 opset）表达：

| 环节 | 所在文件 | 为什么进不了标准 ONNX 图 |
|------|----------|--------------------------|
| TELEA 图像修复 `cv2.inpaint(INPAINT_TELEA)` | `remove_highlight.py` | 基于快速行进法（FMM）的迭代式像素填充，带优先队列与数据依赖的访问顺序，ONNX 没有对应算子，也无法用固定次数的张量运算等价展开 |
| 连通域分析 `cv2.connectedComponentsWithStats` | `highlight_detect.py` | 标签传播类算法，输出数量不定（组件个数），ONNX 图无法表达 |
| 区域分位数自适应阈值 `np.percentile` / `np.partition`（按掩码取动态数量像素） | `highlight_detect.py` | 参与统计的像素集合大小随图像内容变化，是动态 gather + 动态 k 的第 k 大选择；标准算子极难精确表达且无法保证与 NumPy 位级一致 |
| 带提前终止的形态学生长 `_grow_from_core`（dilate 循环 + 收敛判断） | `highlight_detect.py` | 依赖数据的循环终止条件，虽可用 `Loop` 硬编码上限模拟，但与 OpenCV 椭圆核 dilate 的位级一致性无法保证 |
| MediaPipe `.task` 模型包 | `models/face_landmarker.task` | `.task` 是 zip 包（内含 3 个 TFLite 模型 + 元数据），不是单一网络；MediaPipe 图里的前后处理（letterbox、anchor 解码、加权 NMS、旋转裁剪、坐标反投影）在 `.task` 外部，需要另行实现 |
| 双边滤波 `cv2.bilateralFilter`、Canny 边缘 | `remove_highlight.py` | ONNX 无对应算子；理论上可用大量基础算子拼出近似，但无法与 OpenCV 位级一致 |
| 100+ 个 YAML 参数驱动的条件分支（模式选择、ROI 裁剪、面积截断等） | 全流程 | 控制流大量依赖运行时数据（掩码面积、组件数量），不适合静态图 |

**能进 ONNX 的部分**：`.task` 包内的两个神经网络。经解析确认二者**只用标准 TFLite 内建算子，无自定义算子**，可无损转换为 ONNX：

| 模型 | 输入 | 输出 | 算子 |
|------|------|------|------|
| `face_detector.tflite`（BlazeFace 短距人脸检测） | 1×128×128×3，[-1,1] | 896 anchors × 16 回归值 + 896 分数 | CONV_2D、DEPTHWISE_CONV_2D、ADD、CONCATENATION、MAX_POOL_2D、PAD、RELU、RESHAPE、DEQUANTIZE |
| `face_landmarks_detector.tflite`（FaceMesh-V2，478 点） | 1×256×256×3，[0,1] | 1434 = 478×3 坐标 + 人脸存在分数 ×2 | CONV_2D、DEPTHWISE_CONV_2D、ADD、MAX_POOL_2D、PAD、PRELU、LOGISTIC、RESHAPE、DEQUANTIZE |

（`face_blendshapes.tflite` 是表情系数模型，本流程不使用，不转换。）

**用户实际得到的方案**：C++（OpenCV + ONNX Runtime）完整移植——神经网络部分走 ONNX Runtime（由 `.task` 内原始 TFLite 精确转换而来，非第三方替代模型），传统 CV 部分用 OpenCV C++ 以 1:1 相同算法、相同参数移植。C++ 输出与 Python 输出**逐像素高度一致但非位级 100% 相同**，差异来源见第六节（并已被第十节的隔离实验逐项量化）；实际差异用配套脚本量化并如实记录在 `tools/PARITY_RESULTS.md`。

**单文件调用形态（已落地，见第九节）**：在上述 C++ 移植之上，整条流水线进一步封装为**一个**可直接
`session.run` 的 ONNX 文件 `models/high_removal.onnx`。注意这不是「把 TELEA 等塞进了标准算子图」——
那仍然不可能——而是微软官方的「自定义算子封装外部推理运行时」方案
（[add-custom-op.html](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) 之
“Wrapping an external inference runtime in a custom operator”，配套工具
[create_custom_op_wrapper.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/custom_op_wrapper/create_custom_op_wrapper.py)）：
图中只有一个 `ai.high:HighlightRemoval` 自定义算子节点，两个人脸网络与 `default.yaml`
以节点属性内嵌，内核（`libhigh_removal_ops.so`）就是本仓库对拍验证过的 C++ 流水线。

**位级 100% 一致（已落地，见第十节）**：同一个 `models/high_removal.onnx` 另配一个**精确内核**
`libhigh_removal_pyops.so`（默认内核）：Compute 时在宿主进程内直接调用原始 Python 流水线本体
（`highlight_removal.exact_kernel` → `process_image`，MediaPipe + pip OpenCV），因此
`session.run` 输出与 Python `process_image` **逐位相同**（17 张样例实测 17/17 `np.array_equal`，
MAE 0、最大差 0、掩码 IoU 1.0）。约束：宿主必须是 Python 进程且本仓库可导入。
纯 C++ 宿主仍用 C++ 内核（亚像素级浮点尾差如实记录）。

---

## 二、仓库结构与现有 Python 推理链路

```
/workspace
├── app.py / app_simple.py        # Gradio / PyQt UI（本次不动）
├── cli_process.py                # 命令行入口（对齐基准）
├── configs/
│   ├── default.yaml              # 全部默认参数（≈100 项）
│   └── *_presets.yaml            # 灵敏度/强度预设（数值与 default.yaml 一致）
├── highlight_removal/
│   ├── pipeline.py               # 主流程编排（多分辨率折中策略）
│   ├── face_detect.py            # MediaPipe FaceLandmarker + OpenCV 兜底
│   ├── face_landmarks.py         # 478 点拓扑索引 + 几何量
│   ├── face_regions.py           # 区域划分（简化保护模式 / 关键区域模式）
│   ├── skin_mask.py              # 自适应肤色掩码
│   ├── highlight_detect.py       # 高光检测（经典规则 + 区域自适应分位数）
│   ├── remove_highlight.py       # 去高光（LAB 保真压制 + TELEA inpaint 混合）
│   ├── quality_metrics.py        # 质量审计（只产生警告，不影响结果图）
│   └── visualization.py          # 调试可视化（不影响结果图）
├── models/face_landmarker.task   # MediaPipe 模型包（zip：3 个 tflite）
└── data/1.png ... 17.png         # 样例证件照（1280×960 等）
```

`configs/default.yaml` 下的推理链路（`process_scale: compromise`、`refine_upsampled_masks: false`、`enable_alignment: false`、`simple_protect_mode: true`，即 CLI「常用模式」实际路径）：

1. **人脸关键点**：图像缩至 1/4（INTER_AREA）→ MediaPipe FaceLandmarker（`.task`）→ 478 点 ×4 映射回原图；多脸取最大 bbox。
2. **区域划分（1/2 分辨率）**：脸廓多边形（FACE_OVAL 上抬额头）→ 眼/唇凸包膨胀 + 眉弓二次拟合弧带 = 保护区；肤色自适应掩码；可处理皮肤 = 脸 ∩ 皮肤 ∩ 非保护。
3. **高光检测（1/2 分辨率，参数按比例缩放）**：RGB/HSV/LAB/灰度多证据经典规则 + 区域内分位数自适应核心/晕区 + 核心向晕区形态学生长 + 得分截断 + 闭运算/膨胀 + 连通域小面积过滤 + 面积上限截断 → 硬掩码；高斯模糊 × 增益 → 软掩码。
4. **掩码上采样**：硬掩码最近邻、软掩码双线性 + 3×3 高斯。
5. **区域划分（原分辨率）**：同 2，用于修复与保护。
6. **去高光修复（原分辨率，ROI 内）**：软掩码 × Canny(60,130) 边缘保护 → LAB 逐通道 TELEA inpaint（radius 6）与双边滤波（d=11, σc=45, σs=45）参考按 **0.72/0.28** 混合 → 亮度目标 = max(参考+纹理回加, 亮度下限) → L/a/b 按 α 混合；极端核心（L≥188）再对保真结果二次 TELEA inpaint 融合；非皮肤/保护区强制回写原图。
7. 质量审计与可视化——不影响结果图，C++ 不移植。

---

## 三、选定架构

```
┌────────────────────────── C++ 可执行 high_onnx ──────────────────────────┐
│                                                                          │
│  configs/default.yaml ──► yaml-cpp 解析（缺项用与 Python 相同的默认值）    │
│                                                                          │
│  输入图 ──► FaceLandmarkerOrt（ONNX Runtime CPU）                         │
│             ├─ models/face_detector.onnx     （BlazeFace 128×128）        │
│             │   + 自实现：letterbox、anchor 解码、加权 NMS                │
│             └─ models/face_landmarks_detector.onnx（FaceMesh-V2 256×256） │
│                 + 自实现：旋转裁剪、坐标反投影（严格按 MediaPipe 源码）     │
│        ──► face_regions / skin_mask / highlight_detect / remove_highlight │
│             （OpenCV C++ 1:1 移植，相同公式、核尺寸、阈值）                │
│        ──► 输出去高光结果 PNG                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

1. **C++ 库 `high_removal`**（C++17，依赖 OpenCV、ONNX Runtime、yaml-cpp）：1:1 移植推理路径。文件对应：

   | Python | C++ |
   |--------|-----|
   | `face_landmarks.py` 拓扑索引/几何 | `include/high_removal/face_topology.hpp`、`face_geometry.*` |
   | `face_detect.py`（仅 MediaPipe 主路径） | `face_landmarker_ort.*`（ONNX Runtime 重实现 MediaPipe 图） |
   | `face_regions.py` + `skin_mask.py` | `face_regions.*`、`skin_mask.*` |
   | `highlight_detect.py` | `highlight_detect.*` |
   | `remove_highlight.py` | `remove_highlight.*` |
   | `pipeline.py`（compromise 多分辨率路径） | `pipeline.*` |
   | `utils.py`（掩码/缩放/ROI 工具） | `utils.*`（含与 NumPy 语义一致的 percentile/median/partition） |

2. **人脸关键点 ONNX**：用 `tf2onnx --tflite` 把 `.task` 内两个 TFLite **原样**转为 ONNX（`models/face_detector.onnx`、`models/face_landmarks_detector.onnx`），并用 ONNX Runtime 与 MediaPipe 输出做数值对拍。MediaPipe 图的前后处理（`ImageToTensor` letterbox、SSD anchor（4 层 stride 8/16/16/16，共 896）、加权 NMS（IoU 0.3）、眼间旋转对齐、1.5× 方形扩框、landmark 反投影）在 C++ 中按 MediaPipe 源码重实现。**注意**：这不是「转换失败后的替代 FaceMesh」，而是 `.task` 内原始权重的精确转换，语义索引与 Python 完全一致（478 点，无需索引映射）。
3. **C++ CLI**：`high_onnx --input data/1.png --output out.png --config configs/default.yaml`，纯 CPU（与 Python `FORCE_CPU_ONLY=True` 一致），无 GPU 依赖。
4. **对拍脚本** `tools/compare_python_cpp.py`：同一张图分别跑 Python `process_image` 与 C++ CLI，报告结果图 MAE / PSNR / 最大差、逐像素一致率，以及关键点最大偏差与高光掩码 IoU；结果写入 `tools/PARITY_RESULTS.md`。

### 「一个 ONNX 文件」如何做才是对的（已按此落地）

- 把两个网络 + NMS（ONNX 有 `NonMaxSuppression`）+ 裁剪采样拼成一张**纯标准算子**图**理论上可行**，但 TELEA inpaint、连通域、动态分位数仍然进不去——去高光核心必然留在图外。拼一半进图只会把「一次调用」变成「一次调用 + 一堆外部 CV 代码」，反而丢失 MediaPipe 语义的可验证性。**因此本仓库不提供、也不会伪造「纯标准算子」的单图。**
- 正确做法是微软官方的 custom-op wrapper：单节点自定义算子图 + 自定义算子共享库。ONNX 文件承载全部权重与配置（用户只拷一个 onnx），共享库承载不可图化的算法（加载它是 ORT 执行自定义算子的必需步骤，与「装 onnxruntime 才能跑 onnx」同理）。第九节是落地细节。

### 可选延伸（时间允许才做）

`highlight_detect` 中纯张量部分（高斯模糊差、通道阈值打分）理论上可以导出成 `models/highlight_detect.onnx`，但分位数阈值、连通域、生长循环仍在图外，导出物只覆盖不到一半逻辑且对 C++ 集成无增益，**默认不做**；若做，绝不声称该图包含 TELEA inpaint。

---

## 四、对齐范围界定

- **对齐基准**：`configs/default.yaml` 全默认路径（= CLI「常用模式」，预设数值与 default.yaml 相同）。即 `process_scale=compromise`、`refine_upsampled_masks=false`、`enable_alignment=false`、`simple_protect_mode=true`、`mode=混合`、`enable_roi_removal=true`。
- **C++ 支持但不承诺位级一致**：`process_scale` 数值档、`保真`/`强修复` 模式——同样按相同公式移植。
- **不移植**：Gradio/PyQt UI、质量审计文本、调试可视化、OpenCV Haar/LBF 兜底检测器（生产路径始终是 MediaPipe 模型；模型文件缺失时 C++ 直接报错而不是退化到低精度兜底）、`enable_alignment=true` 的轻量对齐分支（默认关闭）、`refine_upsampled_masks=true` 分支（默认关闭）。

## 五、关键一致性细节（移植时逐条核对）

- LAB inpaint 参考混合比 **0.72 / 0.28**；TELEA 半径 `clamp(inpainting_radius,1,9)=6`。
- 双边滤波 `d=11, sigmaColor=45, sigmaSpace=45`；Canny **60/130** + 5×5 高斯；边缘保护强度 0.38。
- 亮度压制 0.94、色度恢复 0.38、纹理保留 0.70（回加系数 0.16×0.70）、最终混合 0.97、亮度下限 0.69、极端核心 +10L。
- 高光检测：局部基线 σ=6 / 大尺度 σ=28 的掩码归一化高斯；色度窗口 31×31；得分权重 0.40/0.22/0.13/0.16/0.12/0.18/0.08；生长上限 16 次（3×3 椭圆核）；闭运算半径 4、膨胀 2、模糊 8、增益 1.85（1/2 分辨率时按 `scaled_highlight_params` 缩放为 2/1/4，min_area 5→1）。
- 区域：额头上抬 `forehead_height_ratio=1.54`、眼保护膨胀 5+1、唇保护 5−2、眉带高 2.30/宽 1.60、肤色平滑半径 4、脸廓收缩 0.018×瞳距。
- NumPy 语义复刻：`np.percentile`（线性插值）、`np.median`（偶数取均值）、`np.partition` 第 k 大阈值（取 `>=kth`，允许并列超额）、`np.polyfit` 二次拟合（双精度最小二乘）。
- OpenCV 调用一律用与 Python 绑定相同的函数与参数（Python `cv2` 本身就是 C++ OpenCV 的绑定，同版本下位级一致）。

## 六、预期偏差来源（诚实声明）

1. **TFLite(XNNPACK) vs ONNX Runtime 浮点尾差**：同一权重不同推理引擎，关键点坐标存在约 1e-3~1e-1 像素级微差；掩码为整数栅格化，绝大多数像素不受影响，但边界个别像素可能翻转，进而通过分位数阈值产生小的连锁差异。
2. **OpenCV 版本差异**：Python 侧 pip OpenCV 与 C++ 侧系统 OpenCV 版本可能不同（算法稳定，但不排除个别函数实现微调）。
3. **NumPy vs 手写统计函数**：按语义复刻，双精度下逐位一致性可期，但 `polyfit`（SVD 最小二乘）与手写正规方程可能有 1e-10 级差异，经栅格化后基本不可见。

因此验收标准定为**量化指标**而非位级相等：结果图 MAE、PSNR、差异像素占比、高光掩码 IoU、478 点最大像素偏差，全部如实记录。

## 七、实施步骤

1. **Phase A**：本计划文档（当前文件）。
2. **Phase B1**：`tools/export_landmarker_onnx.py`——解包 `.task`、tf2onnx 转换、ONNX Runtime vs TFLite/MediaPipe 数值对拍。
3. **Phase B2**：`cpp/` 工程（结构见第三节表格），CMake + OpenCV + ONNX Runtime + yaml-cpp，纯 CPU。
4. **Phase C**：`tools/compare_python_cpp.py` 对 `data/*.png` 全量对拍，结果写 `tools/PARITY_RESULTS.md`；如环境无法编译 C++，如实记录失败原因与已验证部分。
5. 文档：`cpp/README.md`（构建/运行）+ 主 README 链接。

## 八、实测结果（已完成）

以上步骤已全部落地，实测数据如下（细节见 [`tools/PARITY_RESULTS.md`](../tools/PARITY_RESULTS.md)）：

- **ONNX 转换**：`.task` 内两个 TFLite 经 tf2onnx 转出后，随机输入对拍 TFLite 解释器 vs ONNX Runtime，
  最大输出差约 1e-6（浮点尾差量级），确认权重级无损。
- **关键点对拍**：C++/ONNX 关键点链路 vs MediaPipe 原生输出，在原分辨率与 1/4 分辨率
  （流水线实际使用的检测分辨率）上，478 点 xy 最大偏差 **< 0.25 px**，人脸数与 presence 分数一致。
- **端到端对拍（17 张样例图，default.yaml 全默认路径）**：
  平均 MAE **0.0028**/255，平均 PSNR **72.55 dB**，差异像素占比最高 1.05%，
  全部图片最大单像素差 10/255，高光硬掩码平均 IoU **0.9996**。
- 剩余差异全部来自第六节列出的预期来源（推理引擎浮点尾差经整数栅格化在掩码边界放大、两侧 OpenCV 版本不同），
  与算法移植无关。

## 九、单 ONNX 模型交付（已完成）

在 C++ 移植之上，按微软官方 custom-op wrapper 方案交付**一个**可调用的 ONNX 模型：

- **模型**：`models/high_removal.onnx`（约 5.1 MiB，由 `tools/export_high_removal_onnx.py` 生成）。
  图中唯一节点为 `ai.high:HighlightRemoval`；`models/face_detector.onnx`、
  `models/face_landmarks_detector.onnx` 与 `configs/default.yaml` 的完整字节以节点属性
  （uint8 张量，属性名 `face_detector_model` / `face_landmarks_model` / `config_yaml`）内嵌，
  用户无需再携带这三个文件。
- **接口**：输入 `image` uint8 `[H, W, 3]`（BGR，H/W 为 dim_param 动态维）；输出
  `result` uint8 `[H, W, 3]` 与 `hard_mask` uint8 `[H, W]`。未检测到人脸时与 Python 语义一致：
  原样返回输入图、掩码全零。
- **内核**：`cpp/src/custom_op.cpp` → `libhigh_removal_ops.so`，导出 ORT 约定的
  `RegisterCustomOps` 入口；内核反序列化属性（`FaceLandmarkerOrt` 支持内存加载、
  `loadConfigFromString` 解析内嵌 YAML）后调用与 CLI 完全相同的 `hr::processImage`。
  按官方推荐**不链接 libonnxruntime**（`ORT_API_MANUAL_INIT`，OrtApi 由宿主传入），
  因此同一个 .so 可被 Python `onnxruntime`（>= 1.22）或任意 C++ ORT 应用加载；
  链接 `-z nodelete` 防止宿主 dlclose 时 OpenCV 常驻线程执行到已卸载代码。
- **调用**：Python 见 `tools/run_high_onnx.py`（`register_custom_ops_library` + 一次
  `session.run`）；C++ 见 `cpp/tools/high_onnx_session.cpp`。
- **实测**（`tools/compare_python_onnx_session.py --kernel cpp`，17 张全量）：单 ONNX 会话
  （C++ 内核）输出与旧 `high_onnx` CLI 输出**逐位相同**（同一内核代码）；与原始 Python 流程
  平均 MAE **0.0028**/255、平均 PSNR **72.55 dB**、硬掩码平均 IoU **0.9996**、最大单像素差
  10/255——与此前 C++ CLI vs Python 的对拍结果完全一致。**C++ 内核不是位级 100%**，
  差异来源同第六节（第十节给出隔离实测）；位级 100% 由第十节的精确内核提供。
- `onnxruntime-extensions` 提供的额外 CV 算子里没有 TELEA inpaint 与连通域分析，
  对本流程无实质帮助，故未引入该依赖。

## 十、精确内核：位级 100% 一致路径（已完成）

### 10.1 残差来源隔离（先量化，再动手）

`tools/parity_isolate.py` 把「Python(MediaPipe) vs 单 ONNX(C++ 内核)」的 17 张残差拆成独立来源
（完整表格见 `tools/PARITY_RESULTS.md` 隔离章节）：

| 对比 | 隔离对象 | 实测（17 张均值） |
|------|---------|------------------|
| MediaPipe Tasks vs ORT landmarker（1/4 检测分辨率，478 点，换算回原图像素） | 关键点引擎（TFLite/XNNPACK vs ONNX Runtime） | 最大 0.024 px / 平均 0.008 px（单图最大 0.061 px） |
| Python(MediaPipe) vs Python(ORT 关键点)，CV 代码完全相同 | 仅关键点浮点尾差 | MAE 0.0005；掩码 IoU 的全部下降（0.9966~0.9998）都来自这里 |
| Python(ORT 关键点) vs 单 ONNX(C++ 内核)，关键点近似相同 | 仅 OpenCV 版本（pip 4.14 vs apt 4.6）+ C++ 浮点路径 | MAE 0.0026（占总 MAE 0.0028 的绝大部分），掩码 IoU ≈ 1.0000 |
| Python 同图同配置跑两遍 | 流水线自身确定性 | 逐位相同（无随机性） |
| CLI「常用模式」 vs `default.yaml` 原始值（唯一差异 `brow_region_max_fraction` 0.32/0.28） | 配置口径 | 0/17 张输出受影响 |

结论：**MAE 主要来自 OpenCV 版本差异，掩码边界翻转主要来自关键点引擎浮点尾差**。
两者都是「同一算法、不同二进制」的尾差，逐项追平（自编译 pip 同版本 OpenCV、给 C++ 换 TFLite/XNNPACK
推理）仍无法证明位级相同——只要有任何一个浮点路径不同，整数栅格化就可能翻转个别像素。
因此位级 100% 的唯一可靠做法是：让 `session.run` 执行的就是原始 Python 流水线**本体**。

### 10.2 实现：同一个 ONNX 文件，第二个内核库

- **模型不变**：`models/high_removal.onnx` 无需重新导出（图、属性、字节均不变）。
  自定义算子选内核 = 选注册哪个 .so。
- **精确内核** `cpp/src/py_custom_op.cpp` → `libhigh_removal_pyops.so`：注册与 C++ 内核
  完全相同的 `ai.high:HighlightRemoval` 签名；Compute 时经 CPython **稳定 ABI**
  （`Py_LIMITED_API`，Python ≥ 3.10）调用宿主进程内的
  `highlight_removal.exact_kernel.run_from_buffer`，后者用节点属性里内嵌的
  `default.yaml` 构建配置（与 `app_studio.py` 的 `studio_config()` 完全一致）后直接调用
  `highlight_removal.pipeline.process_image`。**没有第二套实现，不存在近似**。
- 仓库定位：优先环境变量 `HIGH_PY_KERNEL_PATH`，否则按 .so 自身路径（`dladdr`）向上推导
  （`cpp/build/` → 仓库根），自动加入 `sys.path`。
- GIL：`sess.run` 在 Python 侧会释放 GIL，内核线程用 `PyGILState_Ensure` 重新获取，
  Python 侧再用可重入锁串行化 MediaPipe 实例访问。
- 刻意**不链接 libpython**：符号由宿主解释器提供。因此纯 C++ 宿主 `dlopen` 本库会因
  符号缺失直接失败——这是预期行为，如实声明：**精确内核只服务 Python 宿主**；
  纯 C++ 部署继续用 C++ 内核。

### 10.3 实测（`tools/compare_python_onnx_session.py`，17 张全量）

| 内核 | 逐位相同（含掩码） | MAE | 最大像素差 | 硬掩码 IoU |
|------|-------------------|-----|-----------|-----------|
| 精确内核 `libhigh_removal_pyops.so`（默认） | **17/17** | **0.0000** | **0** | **1.0000** |
| C++ 快速内核 `libhigh_removal_ops.so` | 0/17 | 0.0028 | 10 | 0.9996 |

`app_studio.py` 的 ONNX / 对比模式默认走精确内核（对比视图 Python−ONNX 恒为 0）；
`HIGH_ONNX_KERNEL=cpp` 切换 C++ 内核，`HIGH_OPS_LIB` 直接指定库文件。

### 10.4 如何调用（一个 ONNX 文件）

```python
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("cpp/build/libhigh_removal_pyops.so")  # 精确内核；C++ 内核换 libhigh_removal_ops.so
sess = ort.InferenceSession("models/high_removal.onnx", so, providers=["CPUExecutionProvider"])
result, hard_mask = sess.run(["result", "hard_mask"], {"image": cv2.imread("data/1.png")})
```

### 10.5 剩余边界（如实声明）

- 精确内核要求：Python（≥ 3.10）宿主 + 本仓库可导入 + `mediapipe`/`opencv-python` 已安装
  + `models/face_landmarker.task` 在仓库内。它是「把原始 Python 流水线装进 ONNX 接口」，
  不是把 Python 依赖消掉。
- 纯 C++ 宿主（无 Python 解释器）没有位级 100% 路径，只有 C++ 内核的亚像素级尾差
  （上表如实记录）；这由第六节的浮点尾差机理决定。
- 位级一致性在「同一台机器、同一套已安装依赖」内成立且可复验（流水线无随机性）；
  跨机器/跨依赖版本时，Python 流水线本身的输出就会随 OpenCV/MediaPipe 版本变化，
  精确内核与 Python 引擎会**一起**变化并保持互相逐位一致。
