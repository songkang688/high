# 与旧产物（git-high2/lib）方法、环境、质量、速度对照

结论先说：

- **方法**：还是以前那套（ONNX 自定义算子 + 静态 OpenCV / yaml-cpp + Linux 开 IPP + Windows mingw-w64 POSIX 交叉 + 同一套浮点开关）。
- **Linux / Windows 环境版本**：与画质/位级对齐有关的版本 **都要对齐**（见下表「必须」列）。这次唯一刻意改的是宿主 **ONNX Runtime 1.22.0**。
- **质量**：去高光 CV 流水线与旧产物一致（注入黄金关键点 **max=0**）。完整链路 17 张里 **14 张位级相同**；另外 3 张是人脸检测浮点噪声，最大差 7/255，和旧 README 里「ONNX vs Python」同一量级。
- **速度**：同一台 4 核 Linux 上热平均 **同量级，没有变慢**；新包大约快 4%～7%。Windows 本云环境不能 `session.Run`，DLL 按旧交叉配方编，未在真机测速。

旧包：`git-high2/lib` + `models/facehi.onnx`，宿主 **onnxruntime 1.29.0**。  
新包：本目录 `output-1.22.0`，宿主 **onnxruntime 1.22.0**。两套不要混用。

## 1. 是不是以前的方法？

是。编译入口仍是 `git-high2/cpp/CMakeLists.txt` + `cmake/mingw-w64-x86_64.cmake`：

| 步骤 | 旧配方 | 本目录 1.22.0 |
|---|---|---|
| 自定义算子 + 静态链 OpenCV / yaml-cpp | 是 | 是 |
| Linux：`g++` Release，`WITH_IPP=ON` | 是 | 是（`/opt/opencv414`） |
| Windows：`x86_64-w64-mingw32-g++-posix` 交叉 | 是（toolchain 写死 posix，不用系统默认 win32） | 是 |
| 编译开关 | `-O2 -ffp-contract=off -fno-fast-math` | 同左（build.ninja 实测） |
| 导出符号 | 仅 `RegisterCustomOps` | 同左 |
| Windows 导入 | 仅 KERNEL32 / msvcrt / ole32 | 同左 |
| strip | `x86_64-w64-mingw32-strip --strip-unneeded` | 同左 |
| 模型 | `onnx/make_facehi_onnx.py` 默认档 | 同脚本，`--out` 写到本目录，不覆盖 `models/` |

唯一不得不改的运行时细节：Python `onnxruntime==1.22.0` 会先占 Default `LoggingManager`，自定义算子里再 `CreateEnv` 会炸。所以人脸检测/关键点子模型改走私有副本 `libfacehi_ort122.so` / `facehi_ort122.dll`（官方 1.22.0）。**去高光 CV 算法、OpenCV、配置、导出图都没换。**

## 2. Linux / Windows 环境版本：哪些必须对齐？

「必须」= 要对齐旧产物画质，或这套 1.22.0 包能跑起来。  
「沿用」= 这次实际用的版本，和旧 README 配方一致，但换补丁级编译器一般不影响。  
「不必钉死」= 只影响本机测速环境，不编进 so/dll。

### 2.1 两端共用（编进产物、影响画质）

| 项 | 旧配方 / 实测 | 本目录 1.22.0 | 是否必须 |
|---|---|---|---|
| OpenCV C++ | 4.14.0 **静态** | 4.14.0 静态 | **必须**（去高光 CV 与 Python/旧 so 数值对齐） |
| 编译开关 | `-O2 -ffp-contract=off -fno-fast-math` | 同左 | **必须** |
| yaml-cpp | 0.8.0 静态 | Linux `0.8.0+dfsg-6build1`；Windows `/opt/xwin/yamlcpp-mingw` 0.8.0 | **必须**（链接）；算法不读 yaml 版本号 |
| C++ 标准 | C++17 | C++17 | **必须**（CMakeLists） |

### 2.2 Linux 编译环境（编 `.so`）

本次实测：Ubuntu 24.04.4 LTS，gcc/g++ **13.3.0**，cmake 3.28.3。

| 项 | 旧 README | 本次实测 | 是否必须 |
|---|---|---|---|
| OpenCV 4.14.0 静态 + **IPP=ON** | 是 | `/opt/opencv414`，`HAVE_IPP` / `HAVE_IPP_ICV`，ICV **2026.0.0** | **必须**（Linux 位级对齐） |
| BUILD_LIST | core,imgproc,imgcodecs,photo,objdetect | 同左 | **必须** |
| g++ | GCC 13 | 13.3.0 | 沿用原方法；主版本不要换 |
| ORT 头文件 | 1.29.0（`GetApi(29)`） | **1.22.0**（`GetApi(22)`） | 新包 **必须 1.22.0**；旧 so **不能**挂 1.22 宿主 |
| 宿主 pip onnxruntime | 1.29.0 | **1.22.0** | 新包 **必须 1.22.0** |
| Python opencv-contrib-python | 4.14.0 | 4.14.0.94 | 对照黄金样本时 **必须**；纯 ONNX 推理不走 Python OpenCV 修图 |
| Python / numpy | 3.12 / 2.4.x | 3.12.3 / 2.5.3 | 不必钉死（解码用，不进 so） |

### 2.3 Windows 交叉编译环境（在 Linux 上编 DLL，与旧 `lib/facehi_custom_ops.dll` 同一条路）

toolchain：`git-high2/cpp/cmake/mingw-w64-x86_64.cmake` 写死 `g++-posix`。系统默认 `x86_64-w64-mingw32-g++` 是 **win32** 线程模型，**不能用来编这套 DLL**。

| 项 | 旧 README | 本次实测 | 是否必须 |
|---|---|---|---|
| 交叉器 | mingw-w64 GCC 13 **POSIX** | `x86_64-w64-mingw32-g++-posix`（包版本 13.2.0-6ubuntu1+26.1，`--version` 为 GCC 13-posix） | **必须** |
| OpenCV | 4.14.0 静态，**IPP=OFF**（mingw 无 IPP） | `/opt/xwin/opencv-mingw` 4.14.0，`HAVE_IPP` 未定义 | **必须**（以前 Windows 就是这样） |
| OpenCV 其它 | WITH_PROTOBUF=OFF、WITH_ADE=OFF，BUILD_LIST 同 Linux | 同左（已装好的 mingw 静态库） | **必须**（与旧 DLL 同一份 OpenCV） |
| yaml-cpp | 0.8.0 静态 | `/opt/xwin/yamlcpp-mingw` 0.8.0 | **必须** |
| ORT 头文件 | win-x64 1.29.0 | win-x64 **1.22.0** | 新包 **必须 1.22.0** |
| 链接 | `-static -static-libgcc -static-libstdc++`，strip | 同左 | **必须**（用户机零 MinGW 运行库） |
| 导出 / 导入 | 仅 `RegisterCustomOps`；KERNEL32/msvcrt/ole32 | `objdump` 核对相同 | **必须** |

> MinGW 无法链 Intel IPP，Windows DLL 与 Linux so 之间个别 OpenCV 原语可能有 ±1/255 舍入差。这是旧包就有的平台差，不是这次引入的。

### 2.4 Windows 运行环境（用户机，本云未跑 `session.Run`）

| 项 | 要求 | 是否必须 |
|---|---|---|
| 系统 | Windows 10/11 x64 | **必须** |
| Python | 3.10+ | **必须** |
| pip `onnxruntime` | **==1.22.0** 官方 win-x64 CPU | **必须**（不要 1.29） |
| 同目录文件 | `facehi.onnx` + `facehi_custom_ops.dll` + `facehi_ort122.dll` | **必须** |
| Visual C++ / MinGW 运行库 | 不需要（静态链好了） | 不必装 |

## 3. 质量（本机 Linux x86-64 实测，跑了两轮，像素结果相同）

注入黄金关键点（隔离人脸引擎，只测 CV 流水线）：

| | 旧 1.29 产物 | 本目录 1.22.0 |
|---|---|---|
| 样本 1 / 13 vs Python 黄金 | max=0，IoU=1.0 | max=0，IoU=1.0 |

完整链路（内置人脸检测）**旧 so+ORT1.29 vs 新 so+ORT1.22**，`data/1.png`…`17.png`：

| 图 | 结果 |
|---|---|
| 3–15、17（共 14 张） | **位级一致**（max=0，IoU=1.0） |
| `1.png` | max=7/255，MAE=0.003183，diff=0.5045%，IoU=0.996630 |
| `2.png` | max=2/255，MAE=0.000001，diff=0.0001%，IoU=1.0 |
| `16.png` | max=3/255，MAE=0.000312，diff=0.0552%，IoU=0.999761 |
| 17 张平均 MAE | 0.000206 |
| 17 张平均 mask IoU | 0.999788 |

`1.png` 这组数字与旧 README / `bench/BENCH_REPORT.md` 里「ONNX vs Python」完全同一量级（MAE 0.0032、max 7/255、diff 0.504%、IoU 0.9966），来源是 ORT 1.29 vs 1.22 做人脸检测时的亚像素关键点噪声，**不是修图配方变了**。

原产物未改：`lib/*.so` 仍 44,345,168 字节，`lib/*.dll` 仍 17,828,383 字节，`models/facehi.onnx` 仍 6,279,805 字节。

## 4. 速度（同一 4 核机器，先后单独跑，避免抢核）

协议与 `bench/BENCH_REPORT.md` 相同：warmup 首张后，17 张热平均，只计 `session.Run`（解码在计时外）。历史 ONNX 热平均 **0.2234 s/张**（ORT 1.29）。

先后单独测两轮（第二轮热平均）：

| | 旧 lib + ORT 1.29 | 本目录 + ORT 1.22 | 相对历史 0.2234 s |
|---|---|---|---|
| 先前一轮 | 0.2260 s/张 | 0.2173 s/张 | 旧吻合；新约快 3% |
| 本轮顺序重测 | 0.2283 s/张 | 0.2129 s/张 | 旧吻合；新约快 5% |
| 冷启动（本轮） | 0.266 s | 0.264 s | 同量级 |

没有变慢。逐张也都是新包略快或持平（本轮比值约 0.84–0.98）。波动来自 4 核云主机，不是算法变重。

Windows DLL：`file` 为 PE32+ x86-64；导出/导入表与旧 DLL 相同。本环境无 Windows，未测 `session.Run` 耗时。
