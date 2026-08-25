#!/usr/bin/env bash
# 导出 git-high1 便携包（三档强度单文件 ONNX + studio 前端 + 调用脚本）到目标目录。
#
# 用法：
#   bash tools/export_git_high1.sh [目标目录]          # 默认 /home/ubuntu/git-high1
#   ONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 bash tools/export_git_high1.sh
#
# 缺 cpp/build/*.so 时自动重新编译（需要 cmake g++ libopencv-dev libyaml-cpp-dev
# python3-dev 与 onnxruntime 官方预编译包）。包内容与 git-high1/README.md 的
# 目录结构一致；不复制 cpp/src、.git、runtime_outputs、venv。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/home/ubuntu/git-high1}"
ORT_ROOT="${ONNXRUNTIME_ROOT:-/opt/ort/onnxruntime-linux-x64-1.22.0}"

EXACT_SO="$REPO/cpp/build/libhigh_removal_pyops.so"
CPP_SO="$REPO/cpp/build/libhigh_removal_ops.so"
if [[ ! -f "$EXACT_SO" || ! -f "$CPP_SO" ]]; then
    echo "[export] cpp/build 缺算子库，开始重建（ONNXRUNTIME_ROOT=$ORT_ROOT）..."
    cmake -S "$REPO/cpp" -B "$REPO/cpp/build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
        -DONNXRUNTIME_ROOT="$ORT_ROOT"
    make -C "$REPO/cpp/build" -j"$(nproc)"
    [[ -f "$EXACT_SO" && -f "$CPP_SO" ]] || {
        echo "[export] 错误：重建后仍缺 .so（精确内核需要 python3-dev）" >&2
        exit 1
    }
fi

echo "[export] 复制到 $DEST ..."
mkdir -p "$DEST/tools" "$DEST/models" "$DEST/configs" "$DEST/data" "$DEST/cpp/build"
cp "$REPO/git-high1/README.md" "$DEST/README.md"
cp "$REPO/app_studio.py" "$DEST/app_studio.py"
cp "$REPO/tools/run_high_onnx.py" "$DEST/tools/run_high_onnx.py"
cp "$REPO/models/high_removal.onnx" \
   "$REPO/models/high_removal_strong.onnx" \
   "$REPO/models/high_removal_daily.onnx" \
   "$REPO/models/high_removal_detail.onnx" \
   "$REPO/models/face_landmarker.task" "$DEST/models/"
cp "$REPO/configs/"*.yaml "$DEST/configs/"
cp "$EXACT_SO" "$CPP_SO" "$DEST/cpp/build/"
# Windows 预编译 DLL（.github/workflows/windows-dll.yml 产物，提交在 cpp/build/windows/）：
# 存在时一并打包，便携包在 Windows 上即插即用（run_high_onnx.py 会自动按平台选 .so/.dll）。
if compgen -G "$REPO/cpp/build/windows/*.dll" > /dev/null; then
    mkdir -p "$DEST/cpp/build/windows"
    cp "$REPO/cpp/build/windows/"*.dll "$DEST/cpp/build/windows/"
fi
cp "$REPO/data/"*.png "$DEST/data/"
rm -rf "$DEST/highlight_removal"
cp -r "$REPO/highlight_removal" "$DEST/highlight_removal"
find "$DEST/highlight_removal" -type d -name __pycache__ -exec rm -rf {} +

echo "[export] 完成：$DEST（$(du -sh "$DEST" | cut -f1)）"
echo "  启动前端：cd $DEST && python app_studio.py            # http://127.0.0.1:7861"
echo "  命令行：  cd $DEST && python tools/run_high_onnx.py -i data/1.png -o out.png"
