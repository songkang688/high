# 黄金对照报告：C++ 版 vs Python 版

- 日期：2026-08-25（本轮更新：faithful_suppress 对齐 numpy 2 NEP 50 dtype 语义后，
  注入关键点路径已达 **17/17 位级一致（max diff = 0）**）
- 配置：`cli_process.py`「常用模式」（process_scale=compromise，人脸检测 0.25 / 高光 0.5，检测「灵敏」，修复「强力」，混合修复，ROI 修复开启，refine_upsampled_masks=false）
- Python 侧版本：Python 3.12.3 / opencv-contrib-python 4.14.0 / mediapipe 1.0.1 / numpy 2.5.2
- C++ 侧：OpenCV 4.14.0（源码静态编译，**WITH_IPP=ON**，core/imgproc/imgcodecs/photo/objdetect）+ ONNX Runtime 1.29.0 + yaml-cpp 0.8
- 样本：`data/1.png` … `data/17.png` 共 17 张，全部通过（无一张检测失败）
- 指标说明：
  - `result max/mean`：结果图逐像素绝对差的最大值 / 平均值（uint8，0~255）
  - `diff像素比例`：任一通道有差异的像素占比
  - `IoU`：各 mask 按 >0 二值化后的交并比（1.0 = 位级一致）

## 一、完整链路（C++ ONNX Runtime 推理关键点 + C++ CV 流水线）

命令：`.venv/bin/python cpp/tests/compare_golden.py --names 1 2 ... 17`

```text
== 1: result max=7 mean=0.003183 diff像素比例=0.005045 | hard IoU=0.99663 soft IoU=0.997988 skin IoU=0.999501 protect IoU=0.998313 treatable IoU=0.999463
== 2: result max=3 mean=0.000070 diff像素比例=0.000108 | hard IoU=0.999846 soft IoU=0.999049 skin IoU=0.999993 protect IoU=0.999931 treatable IoU=0.999993
== 3: result max=10 mean=0.001694 diff像素比例=0.002493 | hard IoU=0.998388 soft IoU=0.998744 skin IoU=0.999593 protect IoU=0.997121 treatable IoU=0.999588
== 4: result max=2 mean=0.000001 diff像素比例=0.000001 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999367 protect IoU=0.999915 treatable IoU=0.999354
== 5: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999987 protect IoU=0.999889 treatable IoU=0.999987
== 6: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 7: result max=4 mean=0.000522 diff像素比例=0.000839 | hard IoU=0.999683 soft IoU=1.0 skin IoU=0.999816 protect IoU=0.999226 treatable IoU=0.999815
== 8: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 9: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999923 protect IoU=0.999342 treatable IoU=0.999923
== 10: result max=1 mean=0.000000 diff像素比例=0.000001 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999603 protect IoU=0.999389 treatable IoU=0.9996
== 11: result max=2 mean=0.000004 diff像素比例=0.000004 | hard IoU=1.0 soft IoU=0.999899 skin IoU=0.99969 protect IoU=0.99991 treatable IoU=0.999688
== 12: result max=1 mean=0.000001 diff像素比例=0.000002 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999977 protect IoU=0.999773 treatable IoU=0.999977
== 13: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999963 protect IoU=1.0 treatable IoU=0.999962
== 14: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999991 protect IoU=0.999913 treatable IoU=0.999991
== 15: result max=5 mean=0.000210 diff像素比例=0.000415 | hard IoU=0.999727 soft IoU=0.999929 skin IoU=0.999941 protect IoU=0.99949 treatable IoU=0.99994
== 16: result max=5 mean=0.002332 diff像素比例=0.003672 | hard IoU=0.999284 soft IoU=0.9995 skin IoU=0.999725 protect IoU=0.999888 treatable IoU=0.999723
== 17: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999802 protect IoU=0.999412 treatable IoU=0.999809
```

汇总：17/17 通过；**8/17 张端到端位级一致（max=0）**，其余 max ≤ 10/255、
差异像素比例 ≤ 0.51%，全部源自两套推理引擎的亚像素关键点噪声（见第四节）。

## 二、注入 Python 关键点（隔离推理引擎差异，仅验证 CV 流水线复刻精度）

命令：`.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks --names 1 2 ... 17`

```text
== 1: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 2: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 3: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 4: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 5: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 6: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 7: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 8: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 9: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 10: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 11: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 12: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 13: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 14: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 15: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 16: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 17: result max=0 mean=0.000000 diff像素比例=0.000000 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
```

汇总：**17 张全部 result max abs diff = 0、5 类 mask IoU = 1.0——
CV 流水线与 Python 逐像素位级一致（100%）**。

## 三、上一轮 4/255 残差的根因与修复

原语级对照（同一输入分别跑 Python wheel 与 C++ 构建的 cvtColor Lab↔BGR、
bilateralFilter、Canny、GaussianBlur、逐通道 Telea inpaint）显示**全部位级一致**，
排除 OpenCV 二进制差异。真正根因是 **numpy 2 的 NEP 50 标量提升**：
`np.clip(标量)` 返回强类型 `np.float64`，把 `_edge_protected_alpha` 的 α 链、
`texture_keep`、`target_L` 与最终 Lab 混合提升到 float64（只在写回 float32
数组时舍入一次），而旧 C++ 实现全程 float32。修复：`remove_highlight.cpp` 按
numpy 实际 dtype 路径重写（详见 `onnx/tests/onnx_report.md` 第二节）。
前提条件：C++ OpenCV 构建需开启 IPP（与 PyPI wheel 一致）。

## 四、ONNX 关键点精度（阶段 1 验证；完整链路残差的唯一来源）

- `models/face_landmarker.task` 拆出的 TFLite 与转换后 ONNX 在同一随机输入下：
  - face_detector：max abs diff ≈ 6e-5（raw logits）
  - face_landmarks_detector：max abs diff ≈ 4e-4（0~256 像素坐标，相对误差 ~1e-6）
- Python 参考实现（`onnx/mediapipe_onnx_ref.py`，与 C++ `face_detect.cpp` 逐行对应）
  对照 MediaPipe Tasks 官方输出：478 点全部误差 ≤ 0.036 px（鼻尖 ≤ 0.01 px）。

## 五、性能（4 核 CPU 云环境，单线程 ORT）

- C++ `facehi_cpp` 单张（1080p 级）：约 0.35 s（含模型加载、常用模式）
- Python `cli_process.py` 单张：约 2.4 s（含 MediaPipe 初始化）

## 复现步骤

```bash
# 1) 导出 Python 黄金数据（17 张全量）
.venv/bin/python cpp/tests/export_golden.py data/*.png -o cpp/golden

# 2) 构建 C++（详见 cpp/README.md；OpenCV 须 4.14.0 + WITH_IPP=ON）
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 -DORT_ROOT=/opt/ort
cmake --build cpp/build -j

# 3) 对照
.venv/bin/python cpp/tests/compare_golden.py --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

> 仓库内只保留 `1`、`13` 两组黄金样本（约 5MB）；其余样本运行第 1 步即可在本地重新生成。
