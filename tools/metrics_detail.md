# 保护细节变体（onnx_detail.yaml）vs default.yaml 实测对比

自动实测生成（2026-08-25，Python 3.12.3 / opencv-contrib-python 4.14.0 / onnxruntime 1.29.0 / mediapipe 1.0.1，Ubuntu 24.04 x64，纯 CPU）。

> 2026-08-25 QA 更新（cursor/qa-onnx-detail-6544）：修复 `scaled_highlight_params`
> 在 1/2 检测分辨率下把本档显式的 `mask_dilate_radius: 0` 强制抬成 1、并把
> `mask_erode_radius: 1` 四舍五入成 0 的问题（详见 `tools/qa_detail.md`）。
> default/daily/strong 输出逐位不变；本档数字微调（≤0.005），下表已按修复后代码重测。

- 变体定位：**保护细节** —— 保留更多皮肤纹理与毛孔，去油光力度弱于 default；输出与 default **不逐位相同**。
- 两侧均为 `highlight_removal.pipeline.process_image`（默认 CPU 运行时 + 关闭可视化），仅配置不同：
  - default 侧：`configs/default.yaml`（= `models/high_removal.onnx` 内嵌配置）
  - detail 侧：`configs/onnx_detail.yaml`（= `models/high_removal_detail.onnx` 内嵌配置）
- 测试图：`data/1.png`、`data/15.png`、`data/16.png`。

## 配置差异（onnx_detail.yaml 相对 default.yaml）

| 参数 | default | onnx_detail |
|------|---------|-------------|
| highlight_removal.mode | 混合 | 保真 |
| highlight_removal.brightness_suppress_strength | 0.94 | 0.62 |
| highlight_removal.chroma_restore_strength | 0.38 | 0.22 |
| highlight_removal.texture_preserve_strength | 0.70 | 0.92 |
| highlight_removal.edge_protect_strength | 0.38 | 0.62 |
| highlight_removal.inpainting_radius | 6 | 2 |
| highlight_removal.poisson_alpha_strength | 0.82 | 0.50 |
| highlight_removal.final_blend_alpha | 0.97 | 0.78 |
| highlight_removal.max_allowed_modify_area_ratio | 0.20 | 0.12 |
| highlight_removal.max_allowed_mean_brightness_change | 22 | 12 |
| highlight_removal.max_allowed_local_color_delta | 15 | 9 |
| highlight_removal.faithful_luminance_floor | 0.69 | 0.88 |
| highlight_removal.extreme_core_extra_l | 10 | 4 |
| highlight_detection.lab_l_threshold | 178 | 198 |
| highlight_detection.adaptive_core_percentile | 90 | 93 |
| highlight_detection.adaptive_halo_percentile | 78 | 85 |
| highlight_detection.adaptive_grow_iterations | 16 | 8 |
| highlight_detection.highlight_max_area_ratio | 0.155 | 0.090 |
| highlight_detection.highlight_min_area | 5 | 12 |
| highlight_detection.soft_mask_gain | 1.85 | 1.40 |
| highlight_detection.mask_dilate_radius | 2 | 0 |
| highlight_detection.mask_erode_radius | 0 | 1 |
| highlight_detection.morph_close_radius | 4 | 2 |
| regions.protect_expand_radius | 5 | 6 |

其余参数与 default.yaml 相同（`simple_protect_mode: true` 保持不变）。

## 实测数字

指标定义（L 为 OpenCV LAB 的 L 通道，0–255）：

- **掩码内 mean |ΔL|**：各自硬掩码（`highlight_mask > 0`）内处理前后 L 的平均绝对变化。越小 = 高光区亮度改动越轻、纹理保得越多。
- **皮肤区 L≥200 下降量**：各自皮肤掩码内 L≥200 像素数「处理前 → 处理后」的减少量。越小 = 去掉的油光越少。（两套配置 protect_expand_radius 不同导致皮肤掩码有几个像素的差异，故「处理前」计数略有出入。）
- **整图 MAE vs 原图**：整图 BGR 平均绝对差。越小 = 整体改动越少。

### data/1.png

| 指标 | default | onnx_detail | detail 更弱？ |
|------|---------|-------------|--------------|
| 硬掩码面积 (px) | 21336 | 12388 | ✅（-42%） |
| 掩码内 mean \|ΔL\| | 8.474 | **2.765** | ✅ 更低 |
| 皮肤区 L≥200 像素 | 17368 → 7545（降 9823） | 17368 → 16408（**降 960**） | ✅ 下降更小 |
| 整图 MAE vs 原图 | 0.1812 | **0.0422** | ✅ 更小 |
| 与 default 输出逐位相同 | — | 否 | ✅ |

### data/15.png

| 指标 | default | onnx_detail | detail 更弱？ |
|------|---------|-------------|--------------|
| 硬掩码面积 (px) | 43924 | 25504 | ✅（-42%） |
| 掩码内 mean \|ΔL\| | 5.969 | **1.919** | ✅ 更低 |
| 皮肤区 L≥200 像素 | 41563 → 25947（降 15616） | 41560 → 39929（**降 1631**） | ✅ 下降更小 |
| 整图 MAE vs 原图 | 0.2027 | **0.0469** | ✅ 更小 |
| 与 default 输出逐位相同 | — | 否 | ✅ |

### data/16.png

| 指标 | default | onnx_detail | detail 更弱？ |
|------|---------|-------------|--------------|
| 硬掩码面积 (px) | 33500 | 19452 | ✅（-42%） |
| 掩码内 mean \|ΔL\| | 11.436 | **3.260** | ✅ 更低 |
| 皮肤区 L≥200 像素 | 49396 → 43950（降 5446） | 49391 → 48956（**降 435**） | ✅ 下降更小 |
| 整图 MAE vs 原图 | 0.2988 | **0.0579** | ✅ 更小 |
| 与 default 输出逐位相同 | — | 否 | ✅ |

## 验收结论

| 验收项 | 要求 | 实测 |
|--------|------|------|
| 掩码内 mean \|ΔL\| 低于 default | ≥ 2/3 张 | **3/3** ✅ |
| 皮肤区 L≥200 下降量小于 default | ≥ 2/3 张 | **3/3** ✅ |
| 整图 MAE vs 原图小于 default | 越小越好 | **3/3** 更小 ✅ |
| 输出不与 default 逐位相同 | 全部 | **3/3** 不同 ✅ |

## 单 ONNX 会话一致性（精确内核）

`models/high_removal_detail.onnx`（内嵌 configs/onnx_detail.yaml）+ 精确内核
`libhigh_removal_pyops.so`，在 `data/1.png` 上：

```text
result    np.array_equal(process_image 结果, session.run 结果)   = True
hard_mask np.array_equal(process_image 掩码, session.run 掩码)   = True
```

即 `session.run` 与 `process_image(onnx_detail.yaml)` **逐位相同**（结果图与硬掩码均一致）。

## 复现方法

```bash
# 1) 导出模型
python tools/export_high_removal_onnx.py --config configs/onnx_detail.yaml \
  --output models/high_removal_detail.onnx

# 2) Python 侧参考输出
#    process_image(img, load_yaml("configs/onnx_detail.yaml") + 默认 CPU 运行时 + 关闭可视化)

# 3) 单 ONNX 会话（精确内核）
#    so.register_custom_ops_library("cpp/build/libhigh_removal_pyops.so")
#    ort.InferenceSession("models/high_removal_detail.onnx", so).run(...)
```
