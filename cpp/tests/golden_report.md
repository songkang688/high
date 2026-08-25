# 黄金对照报告：C++ 版 vs Python 版

- 日期：2026-08-25
- 配置：`cli_process.py`「常用模式」（process_scale=compromise，人脸检测 0.25 / 高光 0.5，检测「灵敏」，修复「强力」，混合修复，ROI 修复开启，refine_upsampled_masks=false）
- Python 侧版本：Python 3.12.3 / opencv-contrib-python 4.14.0 / mediapipe 1.0.1 / numpy 2.5.2
- C++ 侧：OpenCV 4.14.0（源码编译，core/imgproc/imgcodecs/photo/objdetect）+ ONNX Runtime 1.29.0 + yaml-cpp 0.8
- 样本：`data/1.png` … `data/17.png` 共 17 张，全部通过（无一张检测失败）
- 指标说明：
  - `result max/mean`：结果图逐像素绝对差的最大值 / 平均值（uint8，0~255）
  - `diff像素比例`：任一通道有差异的像素占比
  - `IoU`：各 mask 按 >0 二值化后的交并比（1.0 = 位级一致）

## 一、完整链路（C++ ONNX Runtime 推理关键点 + C++ CV 流水线）

命令：`.venv/bin/python cpp/tests/compare_golden.py --names 1 2 ... 17`

```text
== 1: result max=7 mean=0.004568 diff像素比例=0.006839 | hard IoU=0.99663 soft IoU=0.997988 skin IoU=0.999501 protect IoU=0.998313 treatable IoU=0.999463
== 2: result max=4 mean=0.004588 diff像素比例=0.007431 | hard IoU=0.999846 soft IoU=0.999049 skin IoU=0.999993 protect IoU=0.999931 treatable IoU=0.999993
== 3: result max=10 mean=0.004258 diff像素比例=0.006398 | hard IoU=0.998388 soft IoU=0.998744 skin IoU=0.999593 protect IoU=0.997121 treatable IoU=0.999588
== 4: result max=4 mean=0.001625 diff像素比例=0.002346 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999367 protect IoU=0.999915 treatable IoU=0.999354
== 5: result max=4 mean=0.000776 diff像素比例=0.001097 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999987 protect IoU=0.999889 treatable IoU=0.999987
== 6: result max=3 mean=0.001098 diff像素比例=0.001324 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 7: result max=4 mean=0.003923 diff像素比例=0.006240 | hard IoU=0.999683 soft IoU=1.0 skin IoU=0.999816 protect IoU=0.999226 treatable IoU=0.999815
== 8: result max=3 mean=0.002492 diff像素比例=0.004428 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 9: result max=4 mean=0.001435 diff像素比例=0.001951 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999923 protect IoU=0.999342 treatable IoU=0.999923
== 10: result max=4 mean=0.001165 diff像素比例=0.001319 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999603 protect IoU=0.999389 treatable IoU=0.9996
== 11: result max=4 mean=0.000818 diff像素比例=0.000988 | hard IoU=1.0 soft IoU=0.999899 skin IoU=0.99969 protect IoU=0.99991 treatable IoU=0.999688
== 12: result max=4 mean=0.003195 diff像素比例=0.005420 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999977 protect IoU=0.999773 treatable IoU=0.999977
== 13: result max=4 mean=0.002434 diff像素比例=0.004020 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999963 protect IoU=1.0 treatable IoU=0.999962
== 14: result max=4 mean=0.002313 diff像素比例=0.003977 | hard IoU=1.0 soft IoU=1.0 skin IoU=0.999991 protect IoU=0.999913 treatable IoU=0.999991
== 15: result max=5 mean=0.006230 diff像素比例=0.010479 | hard IoU=0.999727 soft IoU=0.999929 skin IoU=0.999941 protect IoU=0.99949 treatable IoU=0.99994
== 16: result max=5 mean=0.006327 diff像素比例=0.009911 | hard IoU=0.999284 soft IoU=0.9995 skin IoU=0.999725 protect IoU=0.999888 treatable IoU=0.999723
== 17: result max=3 mean=0.001094 diff像素比例=0.001408 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=0.999412 treatable IoU=0.999809
```

汇总：17/17 通过；result 最大绝对差 3~10（/255），平均绝对差 ≤ 0.0064，差异像素比例 ≤ 1.05%；
hard mask IoU 最低 0.99663（11/17 张为 1.0），其余 mask IoU 均 ≥ 0.9971。

## 二、注入 Python 关键点（隔离推理引擎差异，仅验证 CV 流水线复刻精度）

命令：`.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks --names 1 2 ... 17`

```text
== 1: result max=4 mean=0.002624 diff像素比例=0.004172 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 2: result max=4 mean=0.004513 diff像素比例=0.007323 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 3: result max=4 mean=0.002921 diff像素比例=0.004982 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 4: result max=4 mean=0.001624 diff像素比例=0.002345 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 5: result max=4 mean=0.000776 diff像素比例=0.001097 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 6: result max=3 mean=0.001098 diff像素比例=0.001324 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 7: result max=4 mean=0.003660 diff像素比例=0.005906 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 8: result max=3 mean=0.002492 diff像素比例=0.004428 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 9: result max=4 mean=0.001435 diff像素比例=0.001951 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 10: result max=4 mean=0.001165 diff像素比例=0.001318 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 11: result max=4 mean=0.000816 diff像素比例=0.000985 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 12: result max=4 mean=0.003194 diff像素比例=0.005418 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 13: result max=4 mean=0.002434 diff像素比例=0.004020 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 14: result max=4 mean=0.002313 diff像素比例=0.003977 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 15: result max=4 mean=0.006092 diff像素比例=0.010218 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 16: result max=4 mean=0.005275 diff像素比例=0.008837 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
== 17: result max=3 mean=0.001094 diff像素比例=0.001408 | hard IoU=1.0 soft IoU=1.0 skin IoU=1.0 protect IoU=1.0 treatable IoU=1.0
```

汇总：17 张图全部 5 类 mask（hard/soft/skin/protect/treatable）IoU = 1.0，即检测/分区/高光判定
整条链路与 Python **位级一致**；结果图残余差异 max ≤ 4/255、平均 ≤ 0.0061、差异像素 ≤ 1.03%。

## 三、残余差异定位

对差异像素做空间归因（`1/5/15` 号样本）：

```text
5:  diff像素=1348   其中硬mask内=489    软环内=859   mask外=0
1:  diff像素=5126   其中硬mask内=3646   软环内=1480  mask外=0
15: diff像素=16951  其中硬mask内=13999  软环内=2952  mask外=0
```

- **所有差异像素都严格位于被修复区域内部（mask 外差异为 0）**，即未处理像素与 Python 位级一致。
- 残余差异来源于修复阶段（`faithful_suppress` / Telea inpaint / 双边滤波参考）的浮点运算：
  Python 侧 numpy 向量化数学库与 PyPI opencv-contrib wheel 的 SIMD/IPP 派发路径，与本地源码编译的
  OpenCV C++ 存在 1 ULP 级别的舍入差，经 Lab↔BGR 往返与 α 混合放大后体现为最多 4/255 的灰度差。
  这是两套二进制之间的固有差异，不是算法逻辑差异（masks IoU=1.0 可证）。

## 四、ONNX 关键点精度（阶段 1 验证）

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

# 2) 构建 C++（详见 cpp/README.md）
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 -DORT_ROOT=/opt/ort
cmake --build cpp/build -j

# 3) 对照
.venv/bin/python cpp/tests/compare_golden.py --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```

> 仓库内只保留 `1`、`13` 两组黄金样本（约 5MB）；其余样本运行第 1 步即可在本地重新生成。
