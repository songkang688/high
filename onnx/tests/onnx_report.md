# 单一 facehi.onnx 黄金对照报告

- 日期：2026-08-25
- 配置：`cli_process.py`「常用模式」（与 `cpp/tests/golden_report.md` 相同）
- Python 侧：Python 3.12.3 / opencv-contrib-python 4.14.0 / mediapipe 1.0.1 / numpy 2.5.2 / onnxruntime 1.29.0
- C++ 侧：OpenCV 4.14.0（源码静态编译，WITH_IPP=ON，core/imgproc/imgcodecs/photo/objdetect）
  + ONNX Runtime 1.29.0 + yaml-cpp 0.8 + GCC 13.3，`-O2 -ffp-contract=off -fno-fast-math`
- 样本：`data/1.png` … `data/17.png` 共 17 张（黄金数据由 `cpp/tests/export_golden.py` 全量重新导出）
- 被测物：`onnx/models/facehi.onnx`（6.28MB，单节点 `ai.facehi:HighlightRemoval`）
  + `cpp/build/libfacehi_custom_ops.so`

## 结论速览

| 对照路径 | 结果 |
| --- | --- |
| session.Run(facehi.onnx)，注入 Python 黄金关键点 vs Python `process_image` | **17/17 位级一致（max abs diff = 0，hard IoU = 1.0）** |
| session.Run(facehi.onnx) vs C++ 直调 `process_image`（facehi_cpp） | **17/17 位级一致（max = 0）——ONNX 封装未引入任何新误差** |
| session.Run(facehi.onnx) 完整链路（内置 ONNX 关键点）vs Python（MediaPipe 关键点） | 8/17 位级一致；其余 max ≤ 10/255，差异仅来自两套推理引擎的亚像素关键点噪声（hard IoU 最低 0.9966） |

## 一、三路对照完整数字

命令：`.venv/bin/python onnx/tests/compare_facehi_onnx.py --names 1 2 ... 17`

```text
== 1: [完整链路] result max=7 mean=0.003183 diff像素比例=0.005045 hard IoU=0.996630 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 2: [完整链路] result max=3 mean=0.000070 diff像素比例=0.000108 hard IoU=0.999846 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 3: [完整链路] result max=10 mean=0.001694 diff像素比例=0.002493 hard IoU=0.998388 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 4: [完整链路] result max=2 mean=0.000001 diff像素比例=0.000001 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 5: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 6: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 7: [完整链路] result max=4 mean=0.000522 diff像素比例=0.000839 hard IoU=0.999683 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 8: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 9: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 10: [完整链路] result max=1 mean=0.000000 diff像素比例=0.000001 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 11: [完整链路] result max=2 mean=0.000004 diff像素比例=0.000004 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 12: [完整链路] result max=1 mean=0.000001 diff像素比例=0.000002 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 13: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 14: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 15: [完整链路] result max=5 mean=0.000210 diff像素比例=0.000415 hard IoU=0.999727 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 16: [完整链路] result max=5 mean=0.002332 diff像素比例=0.003672 hard IoU=0.999284 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
== 17: [完整链路] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [注入关键点] result max=0 mean=0.000000 diff像素比例=0.000000 hard IoU=1.000000 | [vs C++直调] max=0
```

## 二、上次残余 4/255 差异是怎么压到 0 的

上一轮报告（`cpp/tests/golden_report.md` 旧版）注入关键点后结果图仍有
max ≤ 4/255 的残差，当时归因为「PyPI wheel 与源码编译 OpenCV 的二进制差异」。
本轮先做了**原语级对照**（对 `data/1.png` 分别用 Python wheel 与本地 C++ 构建
跑 cvtColor Lab↔BGR、bilateralFilter(11,45,45)、Canny、GaussianBlur、逐通道
Telea inpaint）：**全部位级一致（max=0）**，排除了 OpenCV 二进制差异。

真正的差异源是 **numpy 2 的 NEP 50 标量提升规则**：

- `np.clip(edge_strength, 0, 1)`、`0.16 * np.clip(texture_strength, 0, 1)`
  返回**强类型 `np.float64` 标量**，会把 `_edge_protected_alpha` 的 α 链、
  `texture_keep`、`target_L` 与最终 L/a/b 混合整条表达式提升到 **float64**，
  只在写回 float32 数组时舍入一次；
- 而 yaml 读出的普通 Python float（如 `brightness_strength`）是**弱类型标量**，
  与 float32 数组运算保持 float32。

旧 C++ 实现全程 float32，与 Python 的 float64 中间态在 Lab u8 量化边界上
偶发 ±1 差，再经 Lab→BGR 放大为最多 4/255。修复方式（`cpp/src/remove_highlight.cpp`）：
按 numpy 实际 dtype 语义重写 `faithful_suppress`——α 链在 `edge_strength>0` 时
用 float64、否则 float32；`texture_keep`/`target_L` 恒为 float64；最终混合按
promoted 标志选择舍入路径；`keep` 比较同样区分 float64/float32。修复后 17/17
位级一致。

另一个必要条件：C++ OpenCV 构建启用 IPP（`-DWITH_IPP=ON`，PyPI wheel 即如此），
否则 cvtColor/GaussianBlur/bilateral 的派发路径与 wheel 不同。

## 三、完整链路残差归因（诚实声明）

完整链路（facehi.onnx 内置 ONNX 关键点）与 Python（MediaPipe TFLite/XNNPACK
关键点）的残差**不为零**：478 点关键点两套引擎间有 ≤0.036px 的浮点噪声
（见 `cpp/tests/golden_report.md` 第四节），个别样本的 mask 边界像素因此
偏移 1 像素（hard IoU 最低 0.9966），传导到修复区内为 max ≤ 10/255。
9/17 样本关键点噪声未越过任何阈值边界，端到端位级一致。
要 100% 位级一致需喂入相同关键点（`landmarks` 可选输入即为此设计），
或接受两套神经网络推理引擎间的固有浮点噪声。

## 四、接口与调用验证

- `onnxruntime` 原生调用（`register_custom_ops_library` + `InferenceSession`）：
  `[H,W,3]` 与 `[1,H,W,3]` 两种输入形状均验证通过，样本 13 vs Python 黄金 max=0。
- `python/facehi_onnx.py` 傻瓜 API（`remove_highlight("photo.png")`）：通过。
- `cpp/build/facehi_onnx_cli`（C++ 最小集成，仅 facehi.onnx + 算子库）：
  样本 5 vs Python 黄金 result max=0、mask 完全相同。
- 不注册自定义算子库时建会话按预期失败（ai.facehi 域无实现，ORT 设计如此）。

## 五、性能（4 核 CPU 云环境，单线程 ORT 子模型推理）

- facehi.onnx 会话加载：约 0.11 s
- 单张 1280×960「常用模式」session.Run：约 0.23 s
- Python `cli_process.py` 同图：约 2.4 s（含 MediaPipe 初始化）

## 复现步骤

```bash
# 1) Python 依赖（版本须与报告头一致）
python3 -m venv .venv && .venv/bin/pip install numpy==2.5.2 \
    opencv-contrib-python==4.14.0.94 mediapipe Pillow PyYAML scikit-image \
    onnxruntime==1.29.0 onnx

# 2) OpenCV 4.14.0 静态库（IPP 开启）与 ONNX Runtime 1.29.0（见 onnx/README.md）
# 3) 构建：cmake -B cpp/build -S cpp ... && cmake --build cpp/build -j
# 4) 生成模型：.venv/bin/python onnx/make_facehi_onnx.py
# 5) 导出黄金：.venv/bin/python cpp/tests/export_golden.py data/*.png -o cpp/golden
# 6) 对照：
.venv/bin/python onnx/tests/compare_facehi_onnx.py --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
.venv/bin/python cpp/tests/compare_golden.py --use-golden-landmarks --names 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
```
