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
cmake_minimum_required(VERSION 3.16)
project(facehi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 数值对齐关键：禁止 FMA 收缩与快速数学，保证与 numpy 每一步 float32 舍入一致。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O2 -ffp-contract=off -fno-fast-math)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo objdetect)
find_package(yaml-cpp REQUIRED)

set(ORT_ROOT "" CACHE PATH "ONNX Runtime 解压目录（含 include/ 与 lib/）")
if(NOT ORT_ROOT)
  message(FATAL_ERROR "请设置 -DORT_ROOT=<onnxruntime 目录>")
endif()

add_library(facehi_core
  src/utils.cpp
  src/face_landmarks.cpp
  src/skin_mask.cpp
  src/face_regions.cpp
  src/highlight_detect.cpp
  src/remove_highlight.cpp
  src/quality_metrics.cpp
  src/face_detect.cpp
  src/pipeline.cpp
  src/config.cpp
)
target_include_directories(facehi_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ORT_ROOT}/include
)
target_link_directories(facehi_core PUBLIC ${ORT_ROOT}/lib)
target_link_libraries(facehi_core PUBLIC
  ${OpenCV_LIBS}
  yaml-cpp
  onnxruntime
)

add_executable(facehi_cpp tools/facehi_cli.cpp)
target_link_libraries(facehi_cpp PRIVATE facehi_core)
