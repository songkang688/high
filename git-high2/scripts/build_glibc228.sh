#!/usr/bin/env bash
# 在 glibc 2.28 的系统上重编 Linux 自定义算子（UOS 可加载）。
# Ubuntu 没有 2.28 这个版本：18.04 是 2.27，下一档已是 2.29。
# 实际编译环境用 Debian 10（glibc 2.28，g++ 8.3）。
#
# 先在该系统里装好：
#   OpenCV 4.14.0 静态库，WITH_IPP=ON，BUILD_LIST=core,imgproc,imgcodecs,photo,objdetect
#   yaml-cpp 0.8.0 静态库
#   ONNX Runtime 1.22.0 官方包（只要头文件；libonnxruntime.so.1.22.0 最高 GLIBC_2.27，可直接当 libfacehi_ort122.so）
#
# 产物不再依赖 libstdc++.so.6，动态符号最高 GLIBC_2.27。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cmake -G Ninja -S "$ROOT/git-high2/cpp" -B "$ROOT/git-high2/cpp/build-glibc228" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="${OCV_LINUX:-/opt/opencv414/lib/cmake/opencv4}" \
  -DORT_ROOT="${ORT_LINUX:-/opt/ort-1.22.0}" \
  -DCMAKE_PREFIX_PATH="${YAML_PREFIX:-/opt/yamlcpp}" \
  -DFACEHI_BUILD_TOOLS=OFF \
  -DFACEHI_STATIC_LIBCXX=ON
cmake --build "$ROOT/git-high2/cpp/build-glibc228" -j
strip --strip-unneeded "$ROOT/git-high2/cpp/build-glibc228/libfacehi_custom_ops.so"
echo "OK $ROOT/git-high2/cpp/build-glibc228/libfacehi_custom_ops.so"
