# facehi_daily.onnx —— 日常/平衡档

独立的单文件去高光模型（**日常版**）：去高光够用，不像强力档那么狠，
也不像保护细节档那么弱。适合日常批量处理证件照时的默认选择。

## 基本信息

| 项目 | 值 |
| --- | --- |
| 文件 | `git-high2/models/facehi_daily.onnx` |
| 大小 | 6,283,452 字节（6.28 MB） |
| SHA-256 | `61ad397c1b07f92a657dc2ab16aa75d328fc26849928e58ead71f047b386ad67` |
| 烘焙 mode | `日常模式`（节点属性 `mode` 与 YAML `default_mode` 均为该值，指向独立配置段） |
| 谱系 | facehi（`ai.facehi:HighlightRemoval`），**非** high_removal.onnx |
| 内核 | 与 `facehi.onnx` 完全相同，共用 `lib/libfacehi_custom_ops.so`（Windows 为 `lib/facehi_custom_ops.dll`），未改任何 C++ |
| 生成方式 | `python git-high2/onnx/make_facehi_onnx.py --preset daily`（完整仓库内运行） |

## 烘焙参数：强力档与保护细节档的逐项中值

「日常模式」不是 CLI 既有三档中的任何一档，而是一份独立配置：
高光检测与修复两段中每个数值键取
**强力档（`facehi_strong.onnx`，mode=强力模式）与保护细节档
（`facehi_detail.onnx`，mode=保护细节）两端实际烘焙值的中值**
（整数参数按 0.5 半进位取整）；`face_detection` / `regions` / `pipeline`
三段两端一致，直接沿用（`process_scale=compromise`）。
修复方法保持 `mode: 混合`。

| 参数 | 强力 | **日常** | 细节 |
| --- | --- | --- | --- |
| rgb_brightness_threshold | 192 | **205** | 218 |
| hsv_v_threshold | 178 | **190** | 202 |
| lab_l_threshold | 178 | **188** | 198 |
| hsv_s_upper | 158 | **150** | 142 |
| adaptive_grow_iterations | 16 | **12** | 7 |
| soft_mask_gain | 1.85 | **1.635** | 1.42 |
| adaptive_region_delta / core_delta | 3.4 / 5.8 | **5.1 / 8.3** | 6.8 / 10.8 |
| highlight_min_area / max_area_ratio | 5 / 0.155 | **10 / 0.1225** | 14 / 0.09 |
| brightness_suppress_strength | 0.97 | **0.825** | 0.68 |
| final_blend_alpha | 0.99 | **0.905** | 0.82 |
| poisson_alpha_strength | 0.88 | **0.68** | 0.48 |
| chroma_restore_strength | 0.42 | **0.30** | 0.18 |
| texture_preserve_strength | 0.62 | **0.81** | 1.0 |
| edge_protect_strength | 0.32 | **0.46** | 0.60 |
| inpainting_radius | 7 | **5** | 3 |
| max_allowed_modify_area_ratio | 0.24 | **0.165** | 0.09 |
| max_allowed_mean_brightness_change | 26 | **18** | 10 |
| max_allowed_local_color_delta | 18 | **13** | 8 |
| extreme_core_extra_l | 8 | **19** | 30 |
| faithful_luminance_floor | 0.66 | **0.675** | 0.69 |

其余中值项（检测形态学等）：local_brightness_threshold 6、
local_contrast_threshold 0.026、saturation_pixel_threshold 244、
mask_dilate/erode/blur_radius 1/1/9、oil_shine_s_upper 175、
morph_close_radius 3、forehead/nose_tip/cheek/brow_region_max_fraction
0.285/0.55/0.20/0.30。完整数值见
`git-high2/onnx/make_facehi_onnx.py` 的 `DAILY_DETECTION` / `DAILY_REMOVAL`。

## 用法

```python
import sys; sys.path.insert(0, "git-high2")   # 独立目录内运行则不需要
from facehi_onnx import remove_highlight

remove_highlight("photo.png", variant="daily", save_to="photo_out.png")
```

底层等价于：

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.register_custom_ops_library("git-high2/lib/libfacehi_custom_ops.so")  # 必须
sess = ort.InferenceSession("git-high2/models/facehi_daily.onnx", so)
result, mask = sess.run(None, {"image": bgr_uint8_hwc})   # uint8 BGR [H,W,3]
```

前端 `app_git_high2.py` 左侧「ONNX 模型档位」选
「facehi_daily.onnx · 日常/平衡」即可。

## 验证记录（本分支实测，Ubuntu 24.04 x86-64，onnxruntime 1.29.0 + 仓库自带 so）

对 `data/` 全部 17 张 png，用同一 `lib/libfacehi_custom_ops.so` 分别跑
`facehi.onnx`（常用）、`facehi_daily.onnx`（日常）、`facehi_strong.onnx`（强力）、
`facehi_detail.onnx`（保护细节）完整链路（内置人脸检测）：

1. **与 facehi.onnx（常用）逐位不同**：17/17 张 max abs diff ≥ 2
   （实测 19–34/255，硬标准要求 ≥15/17）。
2. **不贴近强力档**：与 `facehi_strong.onnx` max abs diff ≤ 1 的张数为
   0/17（实测最小 25/255，硬标准要求 <10）。与 `facehi_detail.onnx`
   的 max abs diff 为 19–41/255。
3. **强度有序**：三档硬掩码交集内的 Lab L 均值下降满足
   强力 ≥ 日常 ≥ 细节，17/17 张（硬标准要求 ≥12/17）。
   典型值（1.png）：强力 12.86、日常 7.63、细节 3.49；
   17 张日常档 L 下降范围 5.41–13.11，全部严格落在两端之间。
4. **与 Python 参考对齐**：把同一份中值配置套到
   `highlight_removal.pipeline.process_image` 上，对
   `data/1、3、7、12、15.png` 与 ONNX 对拍：MAE ≤ 0.0009、
   最大像素差 ≤ 6/255、PSNR ≥ 77.09 dB、硬掩码 IoU ≥ 0.9972
   （12.png 逐位一致）——差异仅来自两套人脸推理引擎的亚像素关键点噪声。
5. **内核不变**：与 `facehi.onnx` 内嵌的子模型 / Haar 级联逐字节一致，
   仅 config_yaml（新增「日常模式」段并指向它）与 doc_string 不同；
   `--preset standard` 生成的模型与仓库已提交 `facehi.onnx` 在 mode 与
   config_yaml 上逐字节一致。
