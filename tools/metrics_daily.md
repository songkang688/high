# 日常版（onnx_daily）实测指标

自动生成：`python tools/metrics_daily.py`（2026-08-25）。

- 配置：`configs/onnx_daily.yaml`（日常版 = 当前默认平衡去高光；除顶部注释外与 `configs/default.yaml` 逐字节一致，`highlight_detection` / `highlight_removal` 参数完全相同）
- 模型：`models/high_removal_daily.onnx`（`python tools/export_high_removal_onnx.py --config configs/onnx_daily.yaml --output models/high_removal_daily.onnx`）
- ONNX 侧：`onnxruntime.InferenceSession` + 精确内核 `cpp/build/libhigh_removal_pyops.so`，一次 `session.run` 得到 result / hard_mask
- Python 参考侧：`highlight_removal.pipeline.process_image`，配置 = onnx_daily.yaml + 默认 CPU 运行时 + 关闭可视化（与精确内核对内嵌配置的处理完全一致）
- L = OpenCV LAB 的 L 通道（0~255）；皮肤区 = 流水线肤色掩码 `regions.masks["skin"]`；MAE = 整图三通道平均绝对差（vs 原图）

| 图片 | 硬掩码像素 | 掩码内 mean \|ΔL\| | 掩码内均 L 前→后 | 皮肤 L≥200 前→后 (px) | MAE vs 原图 | session.run ≟ process_image | ≟ default 流水线 | ≟ high_removal.onnx |
|------|-----------|------------------|-----------------|----------------------|-------------|----------------------------|-----------------|--------------------|
| 1.png | 21336 | 8.47 | 206.5→198.1 | 17368→7545 | 0.1812 | ✅ np.array_equal | ✅ | ✅ |
| 15.png | 43924 | 5.97 | 205.3→199.4 | 41563→25947 | 0.2027 | ✅ np.array_equal | ✅ | ✅ |
| 16.png | 33500 | 11.44 | 216.8→205.3 | 49396→43950 | 0.2988 | ✅ np.array_equal | ✅ | ✅ |

- **一致性结论**：3/3 图片上 `session.run(models/high_removal_daily.onnx)`（精确内核 libhigh_removal_pyops.so）与 `process_image(cfg=configs/onnx_daily.yaml)` np.array_equal（result 与 hard_mask 均逐位相同）：**是**。
- **日常版 ≟ 当前默认**：onnx_daily.yaml 与 default.yaml 的配置体（注释外）逐字节一致：**是**；3/3 图片上日常版输出与 default.yaml 流水线及现有 models/high_removal.onnx（同一精确内核）逐位相同：**是**——日常版 = 当前生产平衡效果的具名版本，仅文件名/内嵌配置名不同。

测试环境：Python 3.12.3，onnxruntime 1.29.0，opencv-contrib-python 4.14.0，mediapipe 1.0.1，精确内核 g++ 13.3（Ubuntu 24.04）+ ONNX Runtime 1.22.0 头文件构建。
