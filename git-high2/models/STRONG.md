# facehi_strong.onnx —— 强力版档位说明

三个独立档位模型（强力 / 日常 / 保护细节）中的**强力版**：
**去高光最狠、细节保护最少**。本文件说明烘焙的 mode、生成方式、
与其他两档的预期差异，以及审查/验证记录。

## 烘焙的 mode

- 图中唯一节点 `ai.facehi:HighlightRemoval` 的属性 **`mode="强力模式"`**，
  内嵌配置 YAML 的 `default_mode` 同为 `强力模式`（档位烘焙进图，加载后不可切换）。
- 内嵌 YAML 含四个模式条目：常用模式 / 高保真模式 / 最高质量模式 / **强力模式**；
  运行时只按节点属性 `mode` 取 `强力模式` 这份（已用探针验证，见下）。
- 强力模式 = 现有「常用模式」（检测=**灵敏** + 修复=**强力** + method=**混合** +
  `process_scale=compromise`）基础上加强修复/混合参数。检测（灵敏套）、
  人脸检测、分区、处理分辨率全部与常用模式一致，仅 `highlight_removal` 段有差异：

| 参数 | 常用模式 | 强力模式 | 方向 |
| --- | --- | --- | --- |
| extreme_core_extra_l | 10 | **0** | **关键杠杆**：混合模式的强修复（Telea inpaint）分支从「极亮核心」扩大到硬掩码内全部 L≥lab_l_threshold(178) 的像素——高光被周围皮肤修复填充，而不只是压亮度 |
| brightness_suppress_strength | 0.94 | 1.0 | 亮度压制拉满 |
| poisson_alpha_strength | 0.82 | 0.95 | 强修复混合更重 |
| final_blend_alpha | 0.97 | 1.0 | 最终混合拉满 |
| chroma_restore_strength | 0.38 | 0.46 | 色度回补更强 |
| texture_preserve_strength | 0.70 | 0.55 | 纹理保留更少 |
| edge_protect_strength | 0.38 | 0.28 | 边缘保护更少 |
| inpainting_radius | 6 | 8 | 修复半径更大（kernel 内上限 9） |
| faithful_luminance_floor | 0.69 | 0.62 | 允许把亮度压得更低 |
| max_allowed_modify_area_ratio | 0.20 | 0.28 | 质量守卫上限放宽（守卫只发警告，不回退） |
| max_allowed_mean_brightness_change | 22 | 30 | 同上 |
| max_allowed_local_color_delta | 15 | 20 | 同上 |

算法内核不变：仍是同一套 `ai.facehi` C++ 自定义算子
（`lib/libfacehi_custom_ops.so` / `facehi_custom_ops.dll`），
`faithful_suppress` + `strong_inpaint` 的「混合」路径，无需重新编译算子库。
五官/背景/非皮肤强制回写原图的保护逻辑不受影响（kernel 内硬编码）。

## 如何生成

```bash
python onnx/make_facehi_onnx.py --preset strong --out git-high2/models/facehi_strong.onnx
```

生成器会先用 `cli_process.load_cli_config` 合并三种标准模式（与 Python CLI 完全一致），
再在常用模式之上叠加上表差异项得到「强力模式」，连同两个子模型
（face_detector / face_landmarks_detector）与 Haar 兜底级联一起序列化进节点属性。
文件约 6.28 MB。`python onnx/make_facehi_onnx.py`（无参数）生成的默认模型
与原 `facehi.onnx` 内嵌配置逐字节一致，默认行为未变。

`models/facehi.onnx` **保留原文件未改动**（烘焙 `mode="常用模式"`）；
强力版单独存在于 `facehi_strong.onnx`，未把 facehi.onnx 覆盖成别的档位。

## 与日常版 / 保护细节版的差异

| 档位 | 文件 | 定位 |
| --- | --- | --- |
| **强力**（本文件） | `facehi_strong.onnx` | 灵敏检测 + 强修复分支覆盖整个高光核心：高光压得最低、修复面积最大、纹理与边缘保护最少。适合高光严重、优先「去干净」的照片 |
| 日常 | `facehi_daily.onnx`（另一分支交付） | 高保真取向：去高光与保真折中 |
| 保护细节 | `facehi_detail.onnx`（另一分支交付） | 改动面积最小、纹理边缘保护最多，去高光最轻 |

四档实测（`data/` 全部 17 张，高光掩码区 Lab L 平均下降，越大越狠）：
**强力 10.125** > 默认 8.398 = 日常（当前版）8.398 > 细节 2.325；
**17/17 张强力档均为最大**，且比默认档单张高 0.33–3.85 L。

## 审查/验证记录（本交付实测）

环境：Ubuntu x86-64，onnxruntime 1.29.0 + `lib/libfacehi_custom_ops.so`
（仓库自带，未重编），opencv-python 4.14.0。

1. **烘焙探针（参数确实生效）**：把模型内嵌 YAML 中**只有强力模式段**的
   `brightness_suppress_strength` 改成 0.30 后输出显著变化
   （max pixel diff 68–69）；把**只有常用模式段**改成同样值后输出
   **逐位不变**（max diff = 0）——证明运行时读取的正是烘焙的
   `强力模式` 段，加强参数全部生效。
2. **加载链路**：`find_model("strong")` / `find_model("facehi_strong.onnx")`
   均正确定位；显式 variant 不受环境变量 `FACEHI_ONNX_MODEL` 干扰；
   `FacehiOnnx(variant="strong")` 加载路径正确；缺失档位报中文明确错误；
   `remove_highlight` 会话缓存按档位隔离（default / strong 两个键）。
3. **17/17 张 `session.Run` 成功**：输出均不是原图拷贝
   （高光硬掩码占比 1.25%–2.65%）；强力 vs 默认在掩码区内 MAE
   1.00–4.61（均值 2.17，肉眼可辨），非高光区不受影响。
4. **每张耗时** 0.07–0.5 秒（4 核 CPU，会话热态）。

快速复现：

```python
import sys; sys.path.insert(0, "git-high2")
from facehi_onnx import remove_highlight
remove_highlight("data/1.png", save_to="/tmp/1_strong.png", variant="strong")
```
