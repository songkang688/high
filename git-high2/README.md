# git-high2：效果最好的 ONNX + 本机对比前端

自包含目录。克隆仓库后把本目录整夹拷到本机即可使用：

```bash
# Linux / macOS
cp -a git-high2 ~/git-high2

# Windows（PowerShell）：复制到用户主目录 %USERPROFILE%\git-high2
Copy-Item -Recurse git-high2 "$env:USERPROFILE\git-high2"
```

推荐落地位置：**Linux/macOS `~/git-high2`，Windows `%USERPROFILE%\git-high2`**。

## 为什么选这个 ONNX

`models/facehi.onnx`（约 6.28MB，单节点 `ai.facehi:HighlightRemoval` 自定义算子，
内嵌人脸检测/关键点子模型与三种模式合并配置）。依据 `cursor/detect-audit-0b27`
分支的 DETECT_REPORT 审计结论：

- 默认算法没有被全面更好的替代；注入同一关键点时该 ONNX 与 Python
  参考实现 **max abs diff = 0（17/17 黄金样本位级一致）**。
- 不要改用 `models/high_removal.onnx`：那是另一条谱系，与 Python 对齐更差。
- 不要默认启用 `experimental_better.yaml`：只对纹理优先场景有收益。

## 目录结构

```
git-high2/
├── README.md                 # 本文件
├── models/facehi.onnx        # 唯一对外模型（就是「下载下来的模型」）
├── app_git_high2.py          # 前端：Python / ONNX / 对比 三页签（端口 7862）
├── facehi_onnx.py            # 傻瓜调用：remove_highlight(path) / FacehiOnnx().run(...)
├── requirements.txt          # 运行依赖（版本放宽）
├── lib/                      # 自定义算子库 libfacehi_custom_ops.so（Linux x86-64）
└── cpp/                      # 编译自定义算子库所需的精简源码 + CMakeLists
```

## 快速开始

```bash
cd ~/git-high2                        # 或仓库根目录
pip install -r requirements.txt      # gradio / opencv / numpy / onnxruntime 等
python app_git_high2.py              # 仓库内则：python git-high2/app_git_high2.py
# 打开 http://127.0.0.1:7862
```

前端三个页签：

- **Python 版**：`highlight_removal.pipeline.process_image` 原始流水线
  （常用 / 高保真 / 最高质量三种模式；需要在完整仓库内运行，独立目录时此页
  会提示不可用，ONNX 页不受影响）。
- **ONNX 版**：`models/facehi.onnx` 单会话推理（会话缓存，只加载一次）。
- **对比**：同一份解码数组同时跑两条链路，四宫格展示原图 / Python / ONNX /
  放大差分，并给出 MAE、最大像素差、差异像素占比、PSNR、硬掩码 IoU 与两边耗时。

支持上传图片与 `data/` 样例，中文路径安全（`np.fromfile` 解码、`imencode+tofile` 写盘）。

## 傻瓜调用（不开前端）

```python
import sys; sys.path.insert(0, "git-high2")   # 独立目录内运行则不需要
from facehi_onnx import remove_highlight

out_bgr = remove_highlight("photo.png")                 # 返回 BGR ndarray
remove_highlight("photo.png", save_to="photo_out.png")  # 直接写盘
```

底层等价于：

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.register_custom_ops_library("git-high2/lib/libfacehi_custom_ops.so")  # 必须
sess = ort.InferenceSession("git-high2/models/facehi.onnx", so)
result, mask = sess.run(None, {"image": bgr_uint8_hwc})   # uint8 BGR [H,W,3]
```

## 必须注册自定义算子库（ORT 设计如此）

`facehi.onnx` 的图中只有一个自定义域 `ai.facehi` 的节点，kernel 在
`libfacehi_custom_ops.so`（Windows 为 `facehi_custom_ops.dll`）里。
**不注册该库，onnxruntime 建会话会直接报 `ai.facehi` 域找不到**——
这是 ONNX Runtime 对自定义算子的既定设计，不是缺陷。

库的查找顺序（`facehi_onnx.py`）：环境变量 `FACEHI_ORT_CUSTOM_OPS` →
`git-high2/lib/` → `git-high2/cpp/build/` → 仓库根 `cpp/build/`。

### Linux：编译 `.so`

`lib/` 中若已提供 `libfacehi_custom_ops.so`（Linux x86-64，静态链入
OpenCV 4.14 + yaml-cpp，除系统库外零依赖），直接使用即可。需要重新编译时：

```bash
sudo apt install cmake g++ libyaml-cpp-dev

# ONNX Runtime 官方预编译包（只需头文件，算子库本身不链接 ORT）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz
tar xzf onnxruntime-linux-x64-1.29.0.tgz && sudo mv onnxruntime-linux-x64-1.29.0 /opt/ort

# OpenCV 4.14.0 静态库（数值对齐要求：与 Python 侧 opencv-contrib-python 同版本且开 IPP）
wget https://github.com/opencv/opencv/archive/refs/tags/4.14.0.tar.gz && tar xzf 4.14.0.tar.gz
cmake -B ocv-build -S opencv-4.14.0 -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_IPP=ON \
      -DBUILD_LIST=core,imgproc,imgcodecs,photo,objdetect \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_apps=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv414
cmake --build ocv-build -j && sudo cmake --install ocv-build

# 编译自定义算子库
cmake -B git-high2/cpp/build -S git-high2/cpp -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 -DORT_ROOT=/opt/ort
cmake --build git-high2/cpp/build -j
cp git-high2/cpp/build/libfacehi_custom_ops.so git-high2/lib/
```

### Windows：编译 `.dll`

1. 安装 Visual Studio 2022（含「使用 C++ 的桌面开发」）与 CMake。
2. 下载 ONNX Runtime Windows 预编译包并解压（如 `C:\ort`）：
   `onnxruntime-win-x64-1.29.0.zip`（GitHub microsoft/onnxruntime Releases）。
3. 准备 OpenCV 4.14 静态库（推荐源码编译，配置同上 Linux 的 `-D` 参数；
   `vcpkg install "opencv4[core]" --triplet x64-windows-static` 也可，但要与
   Python 侧 opencv-contrib-python 版本一致才能保证数值对齐）与 yaml-cpp
   （`vcpkg install yaml-cpp --triplet x64-windows-static`）。
4. 在「x64 Native Tools 命令提示符」中执行：

```bat
cmake -B git-high2\cpp\build -S git-high2\cpp -G "Visual Studio 17 2022" -A x64 ^
      -DOpenCV_DIR=C:\opencv414\x64\vc17\staticlib ^
      -DORT_ROOT=C:\ort ^
      -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
      -DVCPKG_TARGET_TRIPLET=x64-windows-static
cmake --build git-high2\cpp\build --config Release
copy git-high2\cpp\build\Release\facehi_custom_ops.dll git-high2\lib\
```

产物 `facehi_custom_ops.dll` 放进 `lib/` 后，`facehi_onnx.py` 与前端会自动找到。

## 模型接口

| 名称 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `image` | 输入 | uint8 | `[H,W,3]` 或 `[1,H,W,3]` | BGR，与 `cv2.imread` 一致 |
| `landmarks` | 输入（可选） | float32 | `[478,3]` | 注入外部关键点则跳过内置检测 |
| `result` | 输出 | uint8 | 同 `image` | 去高光结果 BGR |
| `highlight_mask` | 输出 | uint8 | `[H,W]` | 最终硬掩码 |

模式：内嵌「常用模式」配置。需要其它模式可在仓库内用
`onnx/make_facehi_onnx.py` 改 `mode` 重新生成。

## 不改生产默认算法

本目录只是交付/演示形态：不修改仓库根的 `app.py`、`cli_process.py`、
`configs/default.yaml` 等生产入口的任何默认行为。
