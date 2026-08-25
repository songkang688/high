# 「强力」ONNX 变体去油光强度实测（models/high_removal_strong.onnx）

## 背景

原 `models/high_removal.onnx` 内嵌 `configs/default.yaml`，其 `highlight_removal`
参数与历史「强力」预设完全一致（brightness_suppress 0.94、final_blend 0.97 等），
因此旧「强力」档实际没有任何额外效果。本变体内嵌新的 `configs/onnx_strong.yaml`，
检测与去除均真正加强，输出与 default 模型**不逐位相等**（三张测试图均验证
`outputs_bit_equal=False`）。

## 相对 default.yaml 的参数改动

`highlight_removal`：

| 参数 | default | strong |
| --- | --- | --- |
| brightness_suppress_strength | 0.94 | 0.99 |
| chroma_restore_strength | 0.38 | 0.42 |
| texture_preserve_strength | 0.70 | 0.42 |
| edge_protect_strength | 0.38 | 0.20 |
| inpainting_radius | 6 | 8 |
| poisson_alpha_strength | 0.82 | 0.90 |
| final_blend_alpha | 0.97 | 1.00 |
| max_allowed_modify_area_ratio | 0.20 | 0.28 |
| max_allowed_mean_brightness_change | 22 | 32 |
| max_allowed_local_color_delta | 15 | 22 |
| faithful_luminance_floor | 0.69 | 0.52 |
| extreme_core_extra_l | 10 | 16 |

`highlight_detection`（更激进的检测）：

| 参数 | default | strong |
| --- | --- | --- |
| lab_l_threshold | 178 | 165 |
| adaptive_core_percentile | 90 | 84 |
| adaptive_halo_percentile | 78 | 72 |
| adaptive_grow_iterations | 16 | 22 |
| highlight_max_area_ratio | 0.155 | 0.22 |
| soft_mask_gain | 1.85 | 2.20 |
| mask_dilate_radius | 2 | 3 |
| morph_close_radius | 4 | 5 |

面部保护区域（眼睛/眉毛/嘴唇，`regions.simple_protect_mode: true`）与 default
完全一致，未做任何削弱。

## 测量方法

复现命令：

```bash
python3 tools/compute_strong_metrics.py --images data/1.png data/15.png data/16.png
```

- 两套配置直接从 `models/high_removal.onnx` 与 `models/high_removal_strong.onnx`
  的内嵌 `config_yaml` 属性字节提取（已校验与 `configs/default.yaml` /
  `configs/onnx_strong.yaml` 逐字节一致），并走 `highlight_removal.exact_kernel`
  的同一条配置构建与执行路径。该路径即精确内核（libhigh_removal_pyops）
  `session.run` 的本体，输出逐位相同（见 `highlight_removal/exact_kernel.py`）。
- `mean |ΔL|`：各模型自身硬掩码（`hard_mask` 输出）内，处理前后 LAB L 通道
  （0–255）绝对差均值。越大 = 掩码内压亮越强。
- `皮肤 L≥200 像素数`：流水线皮肤掩码内亮度 L≥200 的像素个数，处理前 → 处理后。
  下降越多 = 反光点消除越彻底。
- `MAE`：全图三通道对原图平均绝对误差（衡量整体改动量）。

## 实测结果（2026-08-25，Python 3.12 / mediapipe 0.10.35 / opencv-contrib 4.x）

### data/1.png（1280×960）

| 指标 | default | strong | 强力更强？ |
| --- | --- | --- | --- |
| 硬掩码像素数 | 21 336 | 30 288 | — |
| mask 内 mean \|ΔL\| | 8.474 | **10.047** | ✔ |
| 皮肤 L≥200：前 → 后 | 17 368 → 7 545（−9 823） | 17 368 → **385**（−16 983） | ✔ |
| MAE vs 原图 | 0.1812 | 0.2954 | — |

### data/15.png（1324×1253）

| 指标 | default | strong | 强力更强？ |
| --- | --- | --- | --- |
| 硬掩码像素数 | 43 924 | 62 344 | — |
| mask 内 mean \|ΔL\| | 5.969 | **6.767** | ✔ |
| 皮肤 L≥200：前 → 后 | 41 563 → 25 947（−15 616） | 41 563 → **12 699**（−28 864） | ✔ |
| MAE vs 原图 | 0.2027 | 0.3175 | — |

### data/16.png（1448×1086）

| 指标 | default | strong | 强力更强？ |
| --- | --- | --- | --- |
| 硬掩码像素数 | 33 500 | 47 552 | — |
| mask 内 mean \|ΔL\| | 11.435 | **11.659** | ✔ |
| 皮肤 L≥200：前 → 后 | 49 396 → 43 946（−5 450） | 49 396 → **34 100**（−15 296） | ✔ |
| MAE vs 原图 | 0.2987 | 0.4247 | — |

## 结论

- 三张图上「强力」的 mask 内 mean |ΔL| 均高于 default，皮肤 L≥200 像素下降量
  均显著更大（1.png −16 983 vs −9 823；15.png −28 864 vs −15 616；
  16.png −15 296 vs −5 450）——满足「至少 2/3 图更强」要求（实际 3/3）。
- 三张图输出与 default 模型均不逐位相等（`outputs_bit_equal=False`）。
- MAE 上升幅度可控（≤0.43/255），且面部保护区域与 default 完全一致。
