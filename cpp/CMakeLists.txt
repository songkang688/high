# 证件照去高光 C++ 移植（OpenCV + ONNX Runtime，纯 CPU）
cmake_minimum_required(VERSION 3.16)
project(high_removal_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo)

# yaml-cpp：优先用系统包；找不到时给出明确报错
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_library(YAML_CPP_LIB yaml-cpp)
  find_path(YAML_CPP_INCLUDE yaml-cpp/yaml.h)
  if(NOT YAML_CPP_LIB OR NOT YAML_CPP_INCLUDE)
    message(FATAL_ERROR "未找到 yaml-cpp，请安装 libyaml-cpp-dev 或设置 CMAKE_PREFIX_PATH")
  endif()
endif()

# ONNX Runtime：通过 -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-x.y.z 指定
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 预编译包根目录")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include" /opt/ort/onnxruntime-linux-x64-1.22.0/include
  PATH_SUFFIXES onnxruntime)
find_library(ORT_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" /opt/ort/onnxruntime-linux-x64-1.22.0/lib)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "未找到 ONNX Runtime，请用 -DONNXRUNTIME_ROOT=... 指定预编译包目录")
endif()

add_library(high_removal STATIC
  src/utils.cpp
  src/config.cpp
  src/geometry.cpp
  src/landmarker.cpp
  src/regions.cpp
  src/highlight.cpp
  src/removal.cpp
  src/pipeline.cpp
)
target_include_directories(high_removal PUBLIC include ${ORT_INCLUDE_DIR})
if(yaml-cpp_FOUND)
  target_link_libraries(high_removal PUBLIC ${OpenCV_LIBS} yaml-cpp ${ORT_LIBRARY})
else()
  target_include_directories(high_removal PUBLIC ${YAML_CPP_INCLUDE})
  target_link_libraries(high_removal PUBLIC ${OpenCV_LIBS} ${YAML_CPP_LIB} ${ORT_LIBRARY})
endif()

add_executable(high_onnx tools/high_cli.cpp)
target_link_libraries(high_onnx PRIVATE high_removal)
