#!/usr/bin/env bash
# 复现 git-high2/output-1.22.0（不覆盖 git-high2/lib 与 git-high2/models）。
# 依赖：g++ cmake ninja mingw-w64、已安装的 OpenCV 4.14 静态库与 ORT 1.22.0 头文件。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CC=gcc CXX=g++
ORT_LINUX="${ORT_LINUX:-/opt/ort-1.22.0}"
ORT_WIN="${ORT_WIN:-/opt/xwin/ort-win/onnxruntime-win-x64-1.22.0}"
OCV_LINUX="${OCV_LINUX:-/opt/opencv414/lib/cmake/opencv4}"
OCV_WIN="${OCV_WIN:-/opt/xwin/opencv-mingw/lib/cmake/opencv4}"

cmake -G Ninja -B "$ROOT/git-high2/cpp/build-ort122" -S "$ROOT/git-high2/cpp" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DOpenCV_DIR="$OCV_LINUX" -DORT_ROOT="$ORT_LINUX" -DFACEHI_BUILD_TOOLS=OFF
cmake --build "$ROOT/git-high2/cpp/build-ort122" -j

export MINGW_FIND_ROOTS=/opt/xwin/opencv-mingw:/opt/xwin/yamlcpp-mingw
cmake -G Ninja -B "$ROOT/git-high2/cpp/build-win-ort122" -S "$ROOT/git-high2/cpp" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ROOT/git-high2/cpp/cmake/mingw-w64-x86_64.cmake" \
  -DOpenCV_DIR="$OCV_WIN" -DORT_ROOT="$ORT_WIN" -DFACEHI_BUILD_TOOLS=OFF
cmake --build "$ROOT/git-high2/cpp/build-win-ort122" -j
x86_64-w64-mingw32-strip --strip-unneeded "$ROOT/git-high2/cpp/build-win-ort122/facehi_custom_ops.dll"

"$ROOT/.venv/bin/python" "$ROOT/onnx/make_facehi_onnx.py" --out /tmp/facehi-1.22.0.onnx
"$ROOT/.venv/bin/python" "$ROOT/git-high2/scripts/pack_output_122.py" \
  --so "$ROOT/git-high2/cpp/build-ort122/libfacehi_custom_ops.so" \
  --dll "$ROOT/git-high2/cpp/build-win-ort122/facehi_custom_ops.dll" \
  --onnx /tmp/facehi-1.22.0.onnx
