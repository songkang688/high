# 在 Linux 上交叉编译 x86_64 Windows 产物的 CMake toolchain。
#
# 依赖：sudo apt install mingw-w64
#
# 必须用 *-posix 变体（POSIX 线程模型）：OpenCV 并行框架与 C++11
# <thread>/<mutex> 在 win32 线程模型的 GCC 下不可用。
# 产物链接时配合 -static（见 CMakeLists.txt）静态链入 winpthread，
# 最终 DLL 除 Windows 系统库外零运行时依赖。
#
# 用法（同一 toolchain 依次编 OpenCV 静态库、yaml-cpp 静态库、facehi）：
#   cmake -B build-win -DCMAKE_TOOLCHAIN_FILE=<本文件> ...
#
# 若依赖安装在 /usr/x86_64-w64-mingw32 之外（例如 /opt/xwin/*-mingw），
# 用冒号分隔的环境变量 MINGW_FIND_ROOTS 追加查找根：
#   export MINGW_FIND_ROOTS=/opt/xwin/opencv-mingw:/opt/xwin/yamlcpp-mingw
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
if(DEFINED ENV{MINGW_FIND_ROOTS})
  string(REPLACE ":" ";" _facehi_extra_roots "$ENV{MINGW_FIND_ROOTS}")
  list(APPEND CMAKE_FIND_ROOT_PATH ${_facehi_extra_roots})
endif()

# 程序（编译器等）在宿主机找，库/头文件/CMake 包只在目标根里找，
# 避免把宿主 Linux 的 OpenCV / yaml-cpp 错链进 Windows DLL。
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
