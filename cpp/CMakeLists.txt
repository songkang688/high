# 证件照去高光 C++ 移植（OpenCV + ONNX Runtime，纯 CPU）
# 支持 Linux（gcc/clang → libhigh_removal_*.so）与 Windows（MSVC → high_removal_*.dll）。
cmake_minimum_required(VERSION 3.16)
project(high_removal_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # 静态库要链接进 libhigh_removal_ops.so
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # 源码为 UTF-8（含中文注释与字符串字面量），不加 /utf-8 时 MSVC 会按本地代码页解析。
  add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo)

# yaml-cpp：优先用系统/工具链包（Ubuntu libyaml-cpp-dev、vcpkg yaml-cpp）；找不到时给出明确报错
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_library(YAML_CPP_LIB yaml-cpp)
  find_path(YAML_CPP_INCLUDE yaml-cpp/yaml.h)
  if(NOT YAML_CPP_LIB OR NOT YAML_CPP_INCLUDE)
    message(FATAL_ERROR "未找到 yaml-cpp，请安装 libyaml-cpp-dev（或 vcpkg install yaml-cpp）或设置 CMAKE_PREFIX_PATH")
  endif()
endif()
# 新版 yaml-cpp（>= 0.8）导出 yaml-cpp::yaml-cpp，旧版导出 yaml-cpp。
if(TARGET yaml-cpp::yaml-cpp)
  set(HR_YAML_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(HR_YAML_TARGET yaml-cpp)
endif()

# ONNX Runtime：通过 -DONNXRUNTIME_ROOT=/path/to/onnxruntime-{linux,win}-x64-x.y.z 指定
# （Linux tgz：lib/libonnxruntime.so；Windows zip：lib/onnxruntime.lib + lib|bin/onnxruntime.dll）
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 预编译包根目录")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include" /opt/ort/onnxruntime-linux-x64-1.22.0/include
  PATH_SUFFIXES onnxruntime)
find_library(ORT_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin" /opt/ort/onnxruntime-linux-x64-1.22.0/lib)
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
# ORT C++ API 走手动初始化（官方自定义算子库做法）：库本身不直接链接 libonnxruntime，
# OrtApi 由可执行文件（Ort::InitApi()）或自定义算子入口（RegisterCustomOps 传入）提供。
target_compile_definitions(high_removal PUBLIC ORT_API_MANUAL_INIT)
if(HR_YAML_TARGET)
  target_link_libraries(high_removal PUBLIC ${OpenCV_LIBS} ${HR_YAML_TARGET})
else()
  target_include_directories(high_removal PUBLIC ${YAML_CPP_INCLUDE})
  target_link_libraries(high_removal PUBLIC ${OpenCV_LIBS} ${YAML_CPP_LIB})
endif()

# 直接调用 C++ 库的 CLI（保留的旧入口）。
add_executable(high_onnx tools/high_cli.cpp)
target_link_libraries(high_onnx PRIVATE high_removal ${ORT_LIBRARY})

# ORT 自定义算子库：models/high_removal.onnx 的 ai.high:HighlightRemoval 内核。
# 只导出 RegisterCustomOps；不链接 libonnxruntime（OrtApi 由宿主传入），
# 因此可被 Python onnxruntime 或任意版本 >= 1.22 的 C++ ORT 应用加载。
# 产物名：Linux libhigh_removal_ops.so / Windows high_removal_ops.dll。
add_library(high_removal_ops SHARED src/custom_op.cpp)
target_link_libraries(high_removal_ops PRIVATE high_removal)
set_target_properties(high_removal_ops PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(WIN32)
  # Windows 用模块定义文件声明唯一导出符号（等价于 Linux 的 version script）。
  target_sources(high_removal_ops PRIVATE src/high_removal_ops.def)
elseif(UNIX AND NOT APPLE)
  target_link_options(high_removal_ops PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/high_removal_ops.version"
    "-Wl,--no-undefined"
    # ORT 在会话选项析构时会 dlclose 本库；nodelete 防止 OpenCV 等依赖连带卸载后
    # 其常驻工作线程执行到已解除映射的代码（进程退出阶段的段错误）。
    "-Wl,-z,nodelete")
endif()

# 精确内核蹦床：同一 ai.high:HighlightRemoval 签名，Compute 时经 CPython C API 调用
# 宿主进程内的 highlight_removal.exact_kernel（原始 Python 流水线本体），
# 因此 session.run 与 Python process_image 逐位相同。
# 只依赖 ORT 头文件 + Python 头文件（Py_LIMITED_API 稳定 ABI，Python >= 3.10）。
# Linux：刻意不链接 libpython——符号由宿主 Python 解释器提供，纯 C++ 宿主无法加载本库（预期行为）。
# Windows：PE 不允许未定义符号，链接稳定 ABI 导入库 python3.lib（运行期解析到宿主的 python3.dll）。
if(WIN32)
  find_package(Python3 COMPONENTS Development.Module Development.SABIModule)
  set(HR_PYOPS_READY "${Python3_Development.SABIModule_FOUND}")
else()
  find_package(Python3 COMPONENTS Development.Module)
  set(HR_PYOPS_READY "${Python3_Development.Module_FOUND}")
endif()
if(HR_PYOPS_READY)
  add_library(high_removal_pyops SHARED src/py_custom_op.cpp)
  target_include_directories(high_removal_pyops PRIVATE ${ORT_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})
  target_compile_definitions(high_removal_pyops PRIVATE ORT_API_MANUAL_INIT)
  set_target_properties(high_removal_pyops PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  if(WIN32)
    target_sources(high_removal_pyops PRIVATE src/high_removal_ops.def)
    target_link_libraries(high_removal_pyops PRIVATE Python3::SABIModule)
  else()
    target_link_libraries(high_removal_pyops PRIVATE ${CMAKE_DL_LIBS})
    target_link_options(high_removal_pyops PRIVATE
      "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/high_removal_ops.version"
      # 不加 --no-undefined：Python C API 符号在运行时由宿主解释器解析。
      "-Wl,-z,nodelete")
  endif()
else()
  message(WARNING "未找到 Python3 开发文件（Linux：python3-dev；Windows：含 libs/python3.lib 的 CPython ≥ 3.10 + CMake ≥ 3.26），跳过精确内核 high_removal_pyops")
endif()

# 单 ONNX 会话示例（新的主要调用方式）：标准 ORT 应用，只依赖 ORT + OpenCV imgcodecs。
add_executable(high_onnx_session tools/high_onnx_session.cpp)
target_include_directories(high_onnx_session PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(high_onnx_session PRIVATE ${ORT_LIBRARY} opencv_core opencv_imgcodecs)
