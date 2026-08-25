# facehi_strong.onnx —— 强力版档位说明

三个独立档位模型（强力 / 日常 / 保护细节）中的**强力版**：
**去高光最狠、细节保护最少**。本文件说明烘焙的 mode、生成方式、
与其他两档的预期差异，以及本交付的验证记录。

## 烘焙的 mode

- 图中唯一节点 `ai.facehi:HighlightRemoval` 的属性 **`mode="强力模式"`**，
  内嵌配置 YAML 的 `default_mode` 同为 `强力模式`（档位烘焙进图，加载后不可切换）。
- 内嵌 YAML 含四个模式条目：常用模式 / 高保真模式 / 最高质量模式 / **强力模式**；
  运行时只按节点属性 `mode` 取 `强力模式` 这份。
- 强力模式 = 现有「常用模式」（检测=**灵敏** + 修复=**强力** + method=**混合** +
  `process_scale=compromise`，即原 facehi.onnx 默认、去高光最狠的档位）
  基础上，把修复/混合参数**再略加强**。检测（灵敏套）、人脸检测、分区、
  处理分辨率全部与常用模式一致，仅 `highlight_removal` 段有差异：

| 参数 | 常用模式 | 强力模式 | 方向 |
| --- | --- | --- | --- |
| brightness_suppress_strength | 0.94 | 0.97 | 亮度压制更强 |
| chroma_restore_strength | 0.38 | 0.42 | 色度回补更强 |
| texture_preserve_strength | 0.70 | 0.62 | 纹理保留更少 |
| edge_protect_strength | 0.38 | 0.32 | 边缘保护更少 |
| inpainting_radius | 6 | 7 | 修复半径更大（kernel 内上限 9） |
| poisson_alpha_strength | 0.82 | 0.88 | 强修复混合更重 |
| final_blend_alpha | 0.97 | 0.99 | 最终混合更接近修复结果 |
| faithful_luminance_floor | 0.69 | 0.66 | 允许把亮度压得更低 |
| extreme_core_extra_l | 10 | 8 | 更多像素进入极亮核心强修复分支 |
| max_allowed_modify_area_ratio | 0.20 | 0.24 | 质量守卫上限放宽（守卫只发警告，不回退） |
| max_allowed_mean_brightness_change | 22 | 26 | 同上 |
| max_allowed_local_color_delta | 15 | 18 | 同上 |

算法内核不变：仍是同一套 `ai.facehi` C++ 自定义算子
（`lib/libfacehi_custom_ops.so` / `facehi_custom_ops.dll`），
`faithful_suppress` + `strong_inpaint` 的「混合」路径，无需重新编译算子库。

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

## 与日常版 / 保护细节版的预期差异

| 档位 | 文件 | 定位 |
| --- | --- | --- |
| **强力**（本文件） | `facehi_strong.onnx` | 检测灵敏套 + 修复强力套再略加强：高光压得最低、修复面积最大、纹理与边缘保护最少。适合高光严重、优先「去干净」的照片 |
| 日常 | 另一分支交付 | 预期以正常检测 + 正常修复为基准（高保真取向）：去高光与保真折中 |
| 保护细节 | 另一分支交付 | 预期以弱检测 / 削弱修复（保真 method）为基准：改动面积最小、纹理边缘保护最多，去高光最轻 |

同一张图三档输出应可肉眼区分：强力版高光区域亮度下降最多、
残留高光最少，但皮肤纹理与边缘细节的保留也最少。

## 验证记录（本交付实测）

环境：Ubuntu x86-64，onnxruntime 1.29.0 + `lib/libfacehi_custom_ops.so`
（仓库自带，未重编），opencv-python 4.14.0。`data/` 全部 17 张
（1.png–17.png）`session.Run` 全部成功：

- 输出均不是原图拷贝（每张 `(result != image).any()` 成立，
  高光硬掩码占比 1.25%–2.65%）。
- 与保留的默认 `facehi.onnx` 对比，**17/17 张**强力版在高光掩码区域的
  Lab L 通道平均下降更大：强力 9.130 vs 默认 8.398（约 +8.7%）；
  单张最大差距如 `data/4.png` 9.547 vs 8.547、`data/11.png` 8.277 vs 7.382。
- 强力 vs 默认全图 MAE 0.011–0.030（/255）：差异集中在高光区域，
  非高光区域不受影响（两档共用同一检测灵敏套与保护回写）。
- 每张耗时约 0.07–0.46 秒（4 核 CPU，会话热态）。

快速复现：

```python
import sys; sys.path.insert(0, "git-high2")
from facehi_onnx import remove_highlight
remove_highlight("data/1.png", save_to="/tmp/1_strong.png", variant="strong")
```
