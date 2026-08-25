# git-high2：效果最好的 ONNX + 本机对比前端

自包含目录。克隆仓库后把本目录整夹拷到本机即可使用：

```bash
# Linux / macOS
cp -a git-high2 ~/git-high2

# Windows（PowerShell）：复制到用户主目录 %USERPROFILE%\git-high2
Copy-Item -Recurse git-high2 "$env:USERPROFILE\git-high2"
```

推荐落地位置：**Linux/macOS `~/git-high2`，Windows `%USERPROFILE%\git-high2`**。

## 模型档位

同一套 `ai.facehi:HighlightRemoval` 自定义算子内核，按烘焙进节点属性的
`mode` 分成互相独立的 onnx 文件（共用同一 `lib/libfacehi_custom_ops.so`）：

| 文件 | 档位 | 烘焙 mode | 特点 |
| --- | --- | --- | --- |
| `models/facehi.onnx` | 常用（强力向） | 常用模式 | 检测=灵敏、修复=强力，默认档 |
| `models/facehi_daily.onnx` | **日常/平衡** | 日常模式（独立中值配置） | 高光检测与修复各参数取强力档与保护细节档两端烘焙值的中值（rgb 阈值 205、修复压制 0.825、final_blend 0.905 等）；method=混合、process_scale=compromise，详见 `models/DAILY.md` |

重新生成：`python git-high2/onnx/make_facehi_onnx.py --preset daily`
（`--preset standard` 对应 facehi.onnx；需在完整仓库内运行）。

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
├── models/facehi.onnx        # 常用档（强力向）对外模型
├── models/facehi_daily.onnx  # 日常/平衡档（烘焙独立「日常模式」中值配置，见 models/DAILY.md）
├── models/DAILY.md           # 日常/平衡档说明与验证记录
├── app_git_high2.py          # 前端：Python / ONNX / 对比 三页签（端口 7862）
├── facehi_onnx.py            # 傻瓜调用：remove_highlight(path) / FacehiOnnx().run(...)
├── onnx/make_facehi_onnx.py  # 模型生成脚本（--preset standard / daily）
├── requirements.txt          # 运行依赖（版本放宽）
├── lib/                      # 自定义算子库：libfacehi_custom_ops.so（Linux x86-64）
│                             #             + facehi_custom_ops.dll（Windows x64）
├── bench/                    # 三引擎平均用时基准（BENCH_REPORT.md + 脚本 + 原始 csv）
└── cpp/                      # 编译自定义算子库所需的精简源码 + CMakeLists
                              #（含 cmake/mingw-w64-x86_64.cmake 交叉编译 toolchain）
```

## 快速开始

```bash
cd ~/git-high2                        # 或仓库根目录
pip install -r requirements.txt      # gradio / opencv / numpy / onnxruntime 等
python app_git_high2.py              # 仓库内则：python git-high2/app_git_high2.py
# 打开 http://127.0.0.1:7862
```

> Linux 无桌面环境（服务器/容器）运行 Python 版时，MediaPipe 需要系统图形库：
> `sudo apt install libegl1 libgl1 libglib2.0-0`（ONNX 版不需要）。

前端三个页签：

- **Python 版**：`highlight_removal.pipeline.process_image` 原始流水线
  （常用 / 高保真 / 最高质量三种模式；需要在完整仓库内运行，独立目录时此页
  会提示不可用，ONNX 页不受影响）。
- **ONNX 版**：`models/facehi.onnx`（常用/强力向）或 `models/facehi_daily.onnx`
  （日常/平衡）单会话推理（左侧「ONNX 模型档位」切换，会话按档位缓存）。
- **对比**：同一份解码数组同时跑两条链路，四宫格展示原图 / Python / ONNX /
  放大差分，并给出 MAE、最大像素差、差异像素占比、PSNR、硬掩码 IoU 与两边耗时。

支持上传图片与 `data/` 样例，中文路径安全（`np.fromfile` 解码、`imencode+tofile` 写盘）。

## 傻瓜调用（不开前端）

```python
import sys; sys.path.insert(0, "git-high2")   # 独立目录内运行则不需要
from facehi_onnx import remove_highlight

out_bgr = remove_highlight("photo.png")                 # 常用档，返回 BGR ndarray
remove_highlight("photo.png", save_to="photo_out.png")  # 直接写盘
remove_highlight("photo.png", variant="daily")          # 日常/平衡档（facehi_daily.onnx）
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

### Windows：已提供 x64 DLL，开箱即用

`lib/facehi_custom_ops.dll`（**x86-64，已随仓库提供**）在 Linux 上用
mingw-w64（GCC 13，POSIX 线程模型）交叉编译：静态链入 OpenCV 4.14.0 与
yaml-cpp 0.8.0，`-O2 -ffp-contract=off -fno-fast-math` 与 Linux so 相同，
除 Windows 系统库（KERNEL32/msvcrt/ole32）外零运行时依赖，无需装
Visual C++ 运行库或 MinGW。唯一导出符号 `RegisterCustomOps`，可被官方
`onnxruntime` win-x64 包（建议 1.22–1.29）的
`SessionOptions.register_custom_ops_library` 直接加载。

使用：把整个 `git-high2` 目录复制到 `%USERPROFILE%\git-high2`
（见文件开头 PowerShell 命令）即可——`facehi_onnx.py` 与前端会自动在
`lib\` 下找到 dll，无需任何配置。若只想单独放置，也可设环境变量
`FACEHI_ORT_CUSTOM_OPS` 指向 dll 的完整路径。

> 数值说明：MinGW 无法链接 Intel IPP（ippicv 只发行 MSVC 版），故 Windows
> DLL 的 OpenCV 为 `WITH_IPP=OFF`。与开 IPP 的 Linux so 相比，个别 OpenCV
> 原语可能有 ±1/255 级别的舍入差；该量级远小于两套人脸推理引擎间的
> 固有差异（见下方验证记录），不影响使用。

#### 在 Linux 上重新交叉编译 dll（本仓库产物的生成方式）

```bash
sudo apt install mingw-w64 cmake
# 1) 用同一 toolchain 静态编 OpenCV 4.14.0（WITH_IPP=OFF、WITH_PROTOBUF=OFF、
#    WITH_ADE=OFF，BUILD_LIST 同 Linux）与 yaml-cpp 0.8.0，
#    分别装到 /opt/xwin/opencv-mingw、/opt/xwin/yamlcpp-mingw；
#    ORT 只需 onnxruntime-win-x64-*.zip 解压出的 include/（不链接 libonnxruntime）。
# 2) 一条命令编 dll：
export MINGW_FIND_ROOTS=/opt/xwin/opencv-mingw:/opt/xwin/yamlcpp-mingw
cmake -B git-high2/cpp/build-win -S git-high2/cpp -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake \
      -DOpenCV_DIR=/opt/xwin/opencv-mingw/lib/cmake/opencv4 \
      -DORT_ROOT=/opt/xwin/ort-win/onnxruntime-win-x64-1.29.0 \
  && cmake --build git-high2/cpp/build-win -j
x86_64-w64-mingw32-strip --strip-unneeded git-high2/cpp/build-win/facehi_custom_ops.dll
cp git-high2/cpp/build-win/facehi_custom_ops.dll git-high2/lib/
```

#### 在 Windows 上用 Visual Studio / vcpkg 重编（可选）

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

产物 `facehi_custom_ops.dll` 放进 `lib/` 后，`facehi_onnx.py` 与前端会自动找到
（`libfacehi_custom_ops.dll` 命名也能识别）。

## 模型接口

| 名称 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `image` | 输入 | uint8 | `[H,W,3]` 或 `[1,H,W,3]` | BGR，与 `cv2.imread` 一致 |
| `landmarks` | 输入（可选） | float32 | `[478,3]` | 注入外部关键点则跳过内置检测 |
| `result` | 输出 | uint8 | 同 `image` | 去高光结果 BGR |
| `highlight_mask` | 输出 | uint8 | `[H,W]` | 最终硬掩码 |

模式：`facehi.onnx` 烘焙「常用模式」，`facehi_daily.onnx` 烘焙独立的
「日常模式」（日常/平衡，强力档与保护细节档的逐项中值配置）。两者接口
完全相同；需要其它档位可在仓库内用
`git-high2/onnx/make_facehi_onnx.py --preset <名称>` 重新生成。

## 验证记录（本交付包实测）

`lib/libfacehi_custom_ops.so` 按上文 Linux 步骤在 Ubuntu 24.04 x86-64 编译
（OpenCV 4.14.0 静态 + IPP，`-ffp-contract=off -fno-fast-math`），实测：

- 注入黄金关键点（`cpp/golden/1_landmarks.txt`）时，ONNX 输出与 Python 黄金结果
  **max abs diff = 0，硬掩码 IoU = 1.0**（位级一致，与 `onnx/tests/onnx_report.md` 结论相同）。
- 完整链路（内置 ONNX 人脸检测）对 `data/1.png`：MAE 0.0032、最大像素差 7/255、
  差异像素占比 0.504%、PSNR 71.58 dB、硬掩码 IoU 0.9966——与基准报告逐位吻合，
  差异仅来自两套推理引擎的亚像素关键点噪声。
- 耗时（4 核 CPU，`data/` 全部 17 张，详见 `bench/BENCH_REPORT.md`）：
  Python 热平均 0.285 秒 / 张，ONNX 热平均 0.223 秒 / 张（快约 21.7%），
  冷启动 ONNX 0.54 秒 vs Python 1.36 秒。

`lib/facehi_custom_ops.dll`（Windows x64）按上文 mingw-w64 交叉编译步骤构建，
已验证：`file` 显示 PE32+ DLL (x86-64)；`x86_64-w64-mingw32-objdump -p`
导出表唯一符号 `RegisterCustomOps`；导入表仅 Windows 系统库
（KERNEL32/msvcrt/ole32）。本云环境无 Windows，未跑过 Windows 上的
`session.Run`；推理数值对拍以 Linux so 为准。

## 不改生产默认算法

本目录只是交付/演示形态：不修改仓库根的 `app.py`、`cli_process.py`、
`configs/default.yaml` 等生产入口的任何默认行为。
