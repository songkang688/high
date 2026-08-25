# facehi.onnx：单一 ONNX 模型封装整条去高光流水线

## 用户最终只需要两个文件

```text
onnx/models/facehi.onnx          # 唯一对外模型文件（约 6.3MB，内嵌一切）
libfacehi_custom_ops.so          # ORT 自定义算子库（构建产物，见下文）
```

`facehi.onnx` 内嵌了：BlazeFace 人脸检测子模型、478 点关键点子模型、
按 `cli_process.py` 合并后的三种模式配置 YAML、Haar 兜底级联 XML。
**不需要**再携带 `configs/`、`models/`、`onnx/models/face_*.onnx`。

## Python 调用

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library("libfacehi_custom_ops.so")   # 必须
sess = ort.InferenceSession("onnx/models/facehi.onnx", so)
result = sess.run(None, {"image": bgr_uint8_hwc})[0]         # uint8 BGR [H,W,3]
```

更傻瓜的封装（自动定位模型与算子库）：

```python
import sys; sys.path.insert(0, "python")
from facehi_onnx import remove_highlight

out_bgr = remove_highlight("photo.png")                      # 返回 BGR ndarray
remove_highlight("photo.png", save_to="photo_out.png")       # 直接写盘
```

## C++ 调用

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "app"};
Ort::SessionOptions so;
so.RegisterCustomOpsLibrary("libfacehi_custom_ops.so");      // 必须
Ort::Session session(env, "facehi.onnx", so);
session.Run(...);  // 输入 "image"，输出 "result" / "highlight_mask"
```

现成 CLI：`cpp/build/facehi_onnx_cli -i data/1.png -o out.png [--mask mask.png]`
（只依赖上述两个文件，自动在可执行文件旁查找算子库）。

## 模型接口

| 名称 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `image` | 输入 | uint8 | `[H,W,3]` 或 `[1,H,W,3]` | BGR，与 `cv2.imread` 一致 |
| `landmarks` | 输入（可选） | float32 | `[478,3]` | 注入外部关键点则跳过内置检测（黄金对照用）；不喂时用图内默认空值 |
| `result` | 输出 | uint8 | 同 `image` | 去高光结果 BGR |
| `highlight_mask` | 输出 | uint8 | `[H,W]`（或 `[1,H,W]`） | 最终硬 mask |

未检测到人脸时与 Python 失败分支一致：`result` 返回原图、mask 全零。

模式：默认「常用模式」（写死在 `mode` 属性）。三种模式的合并配置都已内嵌，
如需其它模式可用 `onnx/make_facehi_onnx.py` 改 `mode` 重新生成。

## 为什么必须带自定义算子库

这不是「标准算子拼出来的纯图」。Telea inpaint、连通域标记、双边滤波、
分位数自适应阈值、MediaPipe 式后处理（加权 NMS、旋转 ROI 裁剪）等步骤
**没有对应的标准 ONNX 算子**，任何声称能用纯标准算子 100% 复现的方案都不成立。

因此采用 ONNX Runtime 官方支持的做法（见
[Custom operators 文档](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
中 "Wrapping an external inference runtime in a custom operator"，官方示例
把整个 OpenVINO 模型序列化进 node attributes）：`facehi.onnx` 图中只有一个
自定义域 `ai.facehi` 的节点 `HighlightRemoval`，kernel 里跑与 Python 逐位对齐的
C++ 流水线。**按 ORT 设计，任何包含自定义域的 ONNX 都必须先注册对应的
kernel 实现库才能建会话**——纯 `onnxruntime` 不调用
`register_custom_ops_library` 会直接报 `ai.facehi` 域找不到。

`libfacehi_custom_ops.so` 静态链入了 OpenCV 与 yaml-cpp，除系统库
（libstdc++/libc/libm）外零运行时依赖，且只导出 `RegisterCustomOps`
一个符号，不会与宿主进程中的其它 OpenCV 副本冲突。

## 构建自定义算子库

```bash
# 依赖：OpenCV 4.14 静态库（-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
#       -DWITH_IPP=ON，模块 core/imgproc/imgcodecs/photo/objdetect）、
#       ONNX Runtime 官方预编译 1.29.0（只需头文件；算子库本身不链接 ORT）、
#       libyaml-cpp-dev
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 -DORT_ROOT=/opt/ort
cmake --build cpp/build -j
# 产物：cpp/build/libfacehi_custom_ops.so、facehi_cpp、facehi_onnx_cli
```

> 数值对齐关键：OpenCV 必须与 Python 侧 `opencv-contrib-python` 同版本（4.14.0）
> 并开启 IPP（PyPI wheel 即如此构建），编译选项 `-ffp-contract=off -fno-fast-math`
> 已写入 CMakeLists。满足这些条件时 17 张黄金样本与 Python **逐像素位级一致**
> （见 `onnx/tests/onnx_report.md`）。

## 生成 / 重新生成 facehi.onnx

```bash
.venv/bin/python onnx/export_onnx.py        # 若子模型缺失，先从 .task 拆出
.venv/bin/python onnx/make_facehi_onnx.py   # 生成 onnx/models/facehi.onnx
```

## 验证

```bash
.venv/bin/python cpp/tests/export_golden.py data/*.png -o cpp/golden   # Python 黄金
.venv/bin/python onnx/tests/compare_facehi_onnx.py --names 1 2 ... 17  # 三路对照
```

完整数字见 [`onnx/tests/onnx_report.md`](tests/onnx_report.md)。

## 性能（4 核 CPU 云环境）

- 会话加载：约 0.11 s；单张 1280×960「常用模式」：约 0.23 s
  （Python `cli_process.py` 同图约 2.4 s）。
