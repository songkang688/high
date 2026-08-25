# 引擎速度对比（SPEED_RESULTS）

自动生成：`python tools/bench_engines.py --runs 3`（2026-08-25）。

- 图片：`data/*.png` 全量 17 张；每个引擎预热 1 次后，每张图计时 3 次取平均；
- 计时范围：`process_image(img, cfg)` / `session.run(...)` 调用本身的墙钟时间（不含 imread/imwrite）；
- 配置：三个引擎完全一致 —— `configs/default.yaml` + 默认 CPU 运行时（`apply_runtime_mode`）+ 关闭可视化，即 `app_studio.py` 工作室 Python 引擎的同一份配置（studio 引擎与 python_orig 调用同一个 `process_image`，无独立实现，故不单列）；
- 环境：Intel(R) Xeon(R) Processor，Linux x86_64，Python 3.12.3，onnxruntime 1.29.0，opencv 4.14.0，mediapipe 0.10.35；
- 算子库：精确内核 `libhigh_removal_pyops.so`，C++ 内核 `libhigh_removal_ops.so`（cpp/ 构建，见 cpp/README.md）；
- 宿主 onnxruntime 版本：精确内核 pip ≥ 1.22 即可；C++ 快速内核在 Python 宿主下需 pip ≥ 1.23（1.22.x 存在嵌套建会话的 LoggingManager 冲突，见 cpp/README.md）。

| 图片 | python_orig (ms) | onnx_exact (ms) | onnx_cpp (ms) | exact/python | cpp/python |
|------|-----------------|-----------------|---------------|--------------|------------|
| 1.png | 294 | 296 | 260 | 1.005 | 0.884 |
| 2.png | 325 | 330 | 298 | 1.015 | 0.917 |
| 3.png | 84 | 85 | 64 | 1.010 | 0.757 |
| 4.png | 311 | 290 | 273 | 0.931 | 0.878 |
| 5.png | 268 | 251 | 181 | 0.938 | 0.675 |
| 6.png | 81 | 79 | 62 | 0.982 | 0.764 |
| 7.png | 344 | 337 | 385 | 0.979 | 1.119 |
| 8.png | 259 | 251 | 186 | 0.968 | 0.719 |
| 9.png | 287 | 288 | 231 | 1.003 | 0.803 |
| 10.png | 277 | 280 | 231 | 1.012 | 0.834 |
| 11.png | 262 | 268 | 196 | 1.021 | 0.748 |
| 12.png | 312 | 308 | 271 | 0.986 | 0.867 |
| 13.png | 298 | 296 | 251 | 0.993 | 0.842 |
| 14.png | 331 | 346 | 247 | 1.043 | 0.747 |
| 15.png | 498 | 499 | 566 | 1.003 | 1.137 |
| 16.png | 422 | 423 | 432 | 1.002 | 1.024 |
| 17.png | 279 | 278 | 224 | 0.995 | 0.801 |

| 引擎 | mean (ms) | median (ms) | 相对 python_orig（mean） |
|------|-----------|-------------|--------------------------|
| python_orig（原始 Python `process_image`，= studio Python 引擎） | 290 | 294 | 1.000 |
| onnx_exact（session.run + 精确内核） | 288 | 290 | 0.994 |
| onnx_cpp（session.run + C++ 快速内核） | 256 | 247 | 0.883 |

## 结论

- **ONNX 精确内核与原始 Python 速度相当**：mean 相差 -0.6%（预期内 —— 精确内核就是原始 Python 流水线本体 + ORT 张量进出拷贝的少量开销，阈值 ±20%）。
- C++ 快速内核 mean 为 python_orig 的 0.883 倍（更快）。

> 以上均为本 Linux 机器实测。Windows DLL 仅做了功能与逐位一致性验证（见 `.github/workflows/windows-dll.yml` 冒烟测试），未在 Windows 上做速度测量。
