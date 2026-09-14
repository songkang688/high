# -*- coding: utf-8 -*-
"""把 ORT 1.22.0 重编产物打进 git-high2/output-1.22.0/，不覆盖 lib/ 与 models/。"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "output-1.22.0"
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MB"


def add_heading_doc(title: str, paragraphs: list[str]) -> Document:
    doc = Document()
    h = doc.add_heading(title, level=0)
    for run in h.runs:
        run.font.size = Pt(18)
    for block in paragraphs:
        p = doc.add_paragraph(block)
        for run in p.runs:
            run.font.size = Pt(11)
            run.font.name = "Calibri"
    return doc


LINUX_DOC = [
    "本目录是 git-high2 默认档（facehi.onnx）按 ONNX / ONNX Runtime 1.22.0 重新导出、"
    "重新编译后的完整交付包。原来的 git-high2/lib/ 与 git-high2/models/ 没有改动。",
    "配套文件：facehi.onnx + libfacehi_custom_ops.so + libfacehi_ort122.so 必须放在同一目录。"
    "模型图里只有自定义域 ai.facehi 的节点，不注册 .so，onnxruntime 会直接报找不到算子。"
    "libfacehi_ort122.so 是 ONNX Runtime 1.22.0 的私有副本，给自定义算子内部跑人脸检测/"
    "关键点子模型用；请勿删除，也不要和 pip 的 onnxruntime 混路径。",
    "环境要求：Linux x86-64，Python 3.10+。",
    "安装依赖：\n  python3 -m venv .venv\n  .venv/bin/pip install -r requirements.txt\n"
    "requirements.txt 已钉死 onnxruntime==1.22.0，请不要换成其他主版本。",
    "推理示例：\n  .venv/bin/python -c \"from facehi_onnx import remove_highlight; "
    "remove_highlight('样例图/1.png', save_to='out.png')\"",
    "或解压 样例图.zip 后：\n  .venv/bin/python -c \"from facehi_onnx import FacehiOnnx; "
    "s=FacehiOnnx(); r,m=s.run('1.png'); print(r.shape, m.shape)\"",
    "底层等价写法：\n  import onnxruntime as ort\n  so = ort.SessionOptions()\n"
    "  so.register_custom_ops_library('libfacehi_custom_ops.so')\n"
    "  sess = ort.InferenceSession('facehi.onnx', so)\n"
    "  result, mask = sess.run(None, {'image': bgr_uint8_hwc})",
    "输入：uint8 BGR [H,W,3]（与 cv2.imread 一致）。输出：result 同形状 BGR，"
    "highlight_mask 为 uint8 [H,W] 硬掩码。",
    "不要把本目录的 .so 与旧的 git-high2/lib/libfacehi_custom_ops.so 混用；"
    "旧库按更高版本 ORT 头文件编过，本目录只保证配合 onnxruntime 1.22.0。",
]

WINDOWS_DOC = [
    "本目录是 git-high2 默认档（facehi.onnx）按 ONNX / ONNX Runtime 1.22.0 重新导出、"
    "重新编译后的完整交付包。原来的 git-high2/lib/ 与 git-high2/models/ 没有改动。",
    "配套文件：facehi.onnx + facehi_custom_ops.dll + facehi_ort122.dll 必须放在同一目录。"
    "模型图里只有自定义域 ai.facehi 的节点，不注册 DLL，onnxruntime 会直接报找不到算子。"
    "facehi_ort122.dll 是 ONNX Runtime 1.22.0 的私有副本，给自定义算子内部跑人脸检测/"
    "关键点子模型用；请勿删除。",
    "环境要求：Windows 10/11 x64，Python 3.10+。DLL 已静态链入 OpenCV / yaml-cpp / libgcc，"
    "一般无需再装 Visual C++ 或 MinGW 运行库。",
    "安装依赖（PowerShell）：\n  python -m venv .venv\n  .venv\\Scripts\\pip install -r requirements.txt\n"
    "请保持 onnxruntime==1.22.0。",
    "推理示例：\n  .venv\\Scripts\\python -c \"from facehi_onnx import remove_highlight; "
    "remove_highlight(r'样例图\\1.png', save_to='out.png')\"",
    "底层等价写法：\n  import onnxruntime as ort\n  so = ort.SessionOptions()\n"
    "  so.register_custom_ops_library('facehi_custom_ops.dll')\n"
    "  sess = ort.InferenceSession('facehi.onnx', so)\n"
    "  result, mask = sess.run(None, {'image': bgr_uint8_hwc})",
    "输入：uint8 BGR [H,W,3]（与 cv2.imread 一致）。输出：result 同形状 BGR，"
    "highlight_mask 为 uint8 [H,W] 硬掩码。",
    "不要把本目录的 DLL 与旧的 git-high2/lib/facehi_custom_ops.dll 混用；"
    "本目录只保证配合官方 onnxruntime 1.22.0 Windows CPU 包。",
]


def write_docs() -> None:
    add_heading_doc("Linux 使用说明（ONNX Runtime 1.22.0）", LINUX_DOC).save(
        str(OUT / "Linux 使用说明.docx"))
    add_heading_doc("Windows 使用说明（ONNX Runtime 1.22.0）", WINDOWS_DOC).save(
        str(OUT / "Windows 使用说明.docx"))
    (OUT / "Linux 使用说明.txt").write_text("\n\n".join(LINUX_DOC) + "\n", encoding="utf-8")
    (OUT / "Windows 使用说明.txt").write_text("\n\n".join(WINDOWS_DOC) + "\n", encoding="utf-8")


def zip_samples() -> None:
    zpath = OUT / "样例图.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(DATA.glob("*.png")):
            zf.write(p, arcname=f"样例图/{p.name}")


def copy_if(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def write_build_info(so: Path, dll: Path, onnx_path: Path, extra: list[str]) -> None:
    files = [
        onnx_path, so, dll,
        OUT / "libfacehi_ort122.so", OUT / "facehi_ort122.dll",
        OUT / "Linux 使用说明.docx", OUT / "Windows 使用说明.docx",
        OUT / "样例图.zip",
    ]
    lines = [
        "git-high2 output-1.22.0",
        f"built_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "onnx_python=1.22.0",
        "onnxruntime=1.22.0 (ORT_API_VERSION 22)",
        "opencv_cxx=4.14.0 static",
        "yaml_cpp=0.8.0 static",
        "original_lib_and_models=kept (not overwritten)",
        "",
        "files:",
    ]
    for p in files:
        if p.is_file():
            lines.append(f"  {p.name}\t{p.stat().st_size}\t{human_size(p.stat().st_size)}\tsha256={sha256(p)}")
    if extra:
        lines.append("")
        lines.extend(extra)
    (OUT / "BUILD_INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = """# git-high2 output-1.22.0

完整交付包（ONNX / ONNX Runtime **1.22.0**）。

`git-high2/lib/` 与 `git-high2/models/` **原产物未删除、未覆盖**。本目录是额外一套。

## 文件

| 文件 | 说明 |
|---|---|
| `facehi.onnx` | 默认档模型，用 Python `onnx==1.22.0` 重新导出 |
| `libfacehi_custom_ops.so` | Linux x86-64 自定义算子（ORT 1.22.0 头文件） |
| `libfacehi_ort122.so` | Linux 私有 ORT 1.22.0，给自定义算子内部子模型用 |
| `facehi_custom_ops.dll` | Windows x64 自定义算子（ORT 1.22.0 头文件） |
| `facehi_ort122.dll` | Windows 私有 ORT 1.22.0，给自定义算子内部子模型用 |
| `Linux 使用说明.docx` / `.txt` | Linux 用法 |
| `Windows 使用说明.docx` / `.txt` | Windows 用法 |
| `样例图.zip` | 仓库 `data/` 17 张样例 |
| `facehi_onnx.py` | 一行调用封装 |
| `requirements.txt` | 钉死 `onnxruntime==1.22.0` |

ONNX 与对应平台的 `.so` / `.dll` **必须配合使用**。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
unzip 样例图.zip
.venv/bin/python -c "from facehi_onnx import remove_highlight; remove_highlight('样例图/1.png', save_to='out.png')"
```

Windows 把 `.venv/bin/` 换成 `.venv\\Scripts\\`，并使用 `facehi_custom_ops.dll`。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=Path, required=True)
    ap.add_argument("--dll", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ort-so", type=Path, required=True,
                    help="官方 libonnxruntime.so.1.22.0，复制为 libfacehi_ort122.so")
    ap.add_argument("--ort-dll", type=Path, required=True,
                    help="官方 onnxruntime.dll，复制为 facehi_ort122.dll")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    copy_if(args.onnx, OUT / "facehi.onnx")
    copy_if(args.so, OUT / "libfacehi_custom_ops.so")
    copy_if(args.dll, OUT / "facehi_custom_ops.dll")
    copy_if(args.ort_so, OUT / "libfacehi_ort122.so")
    copy_if(args.ort_dll, OUT / "facehi_ort122.dll")
    write_docs()
    zip_samples()
    write_readme()
    write_build_info(OUT / "libfacehi_custom_ops.so", OUT / "facehi_custom_ops.dll",
                     OUT / "facehi.onnx", extra=[
                         "inner_ort=libfacehi_ort122.so / facehi_ort122.dll (private ORT 1.22.0)",
                     ])
    print(f"[完成] {OUT}")
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(f"  {p.name:30s} {human_size(p.stat().st_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
