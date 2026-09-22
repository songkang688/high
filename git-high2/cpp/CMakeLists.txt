# facehi C++ 去高光流水线（对齐 Python highlight_removal 包）。
#
# 依赖：
#   - OpenCV >= 4.8（建议 4.14，与 Python 侧 opencv-contrib-python 版本一致，保证数值对齐）
#     需要模块：core imgproc imgcodecs photo objdetect
#   - ONNX Runtime（官方预编译包，ORT_ROOT 指向解压目录）
#   - yaml-cpp（apt: libyaml-cpp-dev）
#
# 构建示例：
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#         -DOpenCV_DIR=/opt/opencv414/lib/cmake/opencv4 \
#         -DORT_ROOT=/opt/onnxruntime
#   cmake --build build -j
#
# Linux 上交叉编译 Windows x64 DLL（mingw-w64，一条命令见 README「交叉编译」节）：
#   cmake -B build-win -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake \
#         -DOpenCV_DIR=<mingw 静态 OpenCV>/lib/cmake/opencv4 \
#         -DORT_ROOT=<onnxruntime-win-x64 解压目录>
cmake_minimum_required(VERSION 3.16)
project(facehi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 数值对齐关键：禁止 FMA 收缩与快速数学，保证与 numpy 每一步 float32 舍入一致。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O2 -ffp-contract=off -fno-fast-math)
endif()

# 在 glibc 2.28 的系统（如 UOS / Debian 10）上加载时打开：
# 用该系统自带的 g++ 编译，并静态链入 libstdc++ / libgcc，
# 避免把宿主机上更高版本的 libstdc++.so.6 带进产物。
option(FACEHI_STATIC_LIBCXX "静态链接 libstdc++ 与 libgcc" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo objdetect)
find_package(yaml-cpp REQUIRED)
# 交叉编译时 yaml-cpp 头文件不在默认 include 路径（Linux 上 /usr/include 侥幸命中），
# 显式找到 yaml-cpp/yaml.h 所在目录。
find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h)

set(ORT_ROOT "" CACHE PATH "ONNX Runtime 解压目录（含 include/ 与 lib/）")
if(NOT ORT_ROOT)
  message(FATAL_ERROR "请设置 -DORT_ROOT=<onnxruntime 目录>")
endif()

# facehi_core 会被静态链入 facehi_custom_ops 共享库，必须 PIC。
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(facehi_core
  src/utils.cpp
  src/face_landmarks.cpp
  src/skin_mask.cpp
  src/face_regions.cpp
  src/highlight_detect.cpp
  src/remove_highlight.cpp
  src/quality_metrics.cpp
  src/face_detect.cpp
  src/inner_ort.cpp
  src/pipeline.cpp
  src/config.cpp
)
target_include_directories(facehi_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ORT_ROOT}/include
)
if(YAML_CPP_INCLUDE_DIR)
  target_include_directories(facehi_core PUBLIC ${YAML_CPP_INCLUDE_DIR})
endif()
if(WIN32)
  # 静态链 yaml-cpp 时必须定义，否则头文件按 __declspec(dllimport) 生成
  # __imp_ 前缀符号导致链接失败。
  target_compile_definitions(facehi_core PUBLIC YAML_CPP_STATIC_DEFINE)
endif()
# ORT_API_MANUAL_INIT：facehi_core 不直接依赖 libonnxruntime 符号。
# - 独立可执行文件（facehi_cpp）自行链接 onnxruntime 并在 main 里 Ort::InitApi()；
# - 自定义算子库（facehi_custom_ops）不链接 onnxruntime，在 RegisterCustomOps
#   中用宿主传入的 OrtApiBase 初始化（ORT 官方自定义算子库约定，避免同进程
#   出现两份运行时）。
target_compile_definitions(facehi_core PUBLIC ORT_API_MANUAL_INIT)
target_link_directories(facehi_core PUBLIC ${ORT_ROOT}/lib)
# 优先静态 yaml-cpp，使 facehi_custom_ops 除系统库外零运行时依赖。
find_library(YAML_CPP_STATIC NAMES libyaml-cpp.a)
if(YAML_CPP_STATIC)
  set(FACEHI_YAML_CPP ${YAML_CPP_STATIC})
else()
  set(FACEHI_YAML_CPP yaml-cpp)
endif()
target_link_libraries(facehi_core PUBLIC
  ${OpenCV_LIBS}
  ${FACEHI_YAML_CPP}
)

# 交叉编译 DLL 时只需要 facehi_custom_ops：两个演示 CLI 需要链接
# libonnxruntime（MSVC 导入库），跳过以保持 mingw 交叉构建极简。
if(CMAKE_CROSSCOMPILING)
  set(_facehi_tools_default OFF)
else()
  set(_facehi_tools_default ON)
endif()
option(FACEHI_BUILD_TOOLS "构建 facehi_cpp / facehi_onnx_cli 演示可执行文件" ${_facehi_tools_default})

if(FACEHI_BUILD_TOOLS)
  add_executable(facehi_cpp tools/facehi_cli.cpp)
  target_link_libraries(facehi_cpp PRIVATE facehi_core onnxruntime)
endif()

# ORT 自定义算子共享库：唯一对外符号是 RegisterCustomOps。
# 静态链入 facehi_core / OpenCV / yaml-cpp 后用 --exclude-libs 隐藏内部符号，
# 避免与宿主进程中的其他 OpenCV 副本冲突。
add_library(facehi_custom_ops SHARED src/custom_op.cpp)
target_link_libraries(facehi_custom_ops PRIVATE facehi_core ${CMAKE_DL_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(facehi_custom_ops PRIVATE "LINKER:--exclude-libs,ALL")
endif()
# GCC 8 的 std::filesystem 在独立的 libstdc++fs 里。
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  if(FACEHI_STATIC_LIBCXX)
    target_link_libraries(facehi_custom_ops PRIVATE "-Wl,-Bstatic" stdc++fs "-Wl,-Bdynamic")
  else()
    target_link_libraries(facehi_custom_ops PRIVATE stdc++fs)
  endif()
endif()
if(FACEHI_STATIC_LIBCXX AND NOT WIN32)
  target_link_options(facehi_custom_ops PRIVATE -static-libstdc++ -static-libgcc)
endif()
if(WIN32)
  # facehi_onnx.py 按 "facehi_custom_ops.dll" 查找：去掉 MinGW 默认 lib 前缀。
  set_target_properties(facehi_custom_ops PROPERTIES PREFIX "")
  # .def 双保险：与 custom_op.cpp 的 __declspec(dllexport) 一致，
  # 保证导出名为无修饰的 RegisterCustomOps。
  target_sources(facehi_custom_ops PRIVATE src/facehi_custom_ops.def)
  if(MINGW)
    # 自包含 DLL：静态链入 libgcc / libstdc++ / winpthread，
    # 用户机器无需再装 MinGW 运行时。
    target_link_options(facehi_custom_ops PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endif()

if(FACEHI_BUILD_TOOLS)
  # 单一 ONNX 入口的 CLI：只依赖 facehi.onnx + facehi_custom_ops，
  # 演示用户侧最小集成（Ort::Session + RegisterCustomOpsLibrary + Run）。
  add_executable(facehi_onnx_cli tools/facehi_onnx_cli.cpp)
  target_include_directories(facehi_onnx_cli PRIVATE ${ORT_ROOT}/include)
  target_link_directories(facehi_onnx_cli PRIVATE ${ORT_ROOT}/lib)
  target_link_libraries(facehi_onnx_cli PRIVATE
    opencv_imgcodecs opencv_core
    onnxruntime
  )
endif()
