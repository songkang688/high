# git-high2 output-1.22.0

完整交付包（ONNX / ONNX Runtime **1.22.0**）。

`git-high2/lib/` 与 `git-high2/models/` **原产物未删除、未覆盖**。本目录是额外一套。

## 文件

| 文件 | 说明 |
|---|---|
| `facehi.onnx` | 默认档模型，用 Python `onnx==1.22.0` 重新导出 |
| `libfacehi_custom_ops.so` | Linux x86-64 自定义算子。按 glibc 2.28 重编，libstdc++ 静态链入，UOS 可加载 |
| `libfacehi_ort122.so` | Linux 私有 ORT 1.22.0，给自定义算子内部子模型用 |
| `facehi_custom_ops.dll` | Windows x64 自定义算子（ORT 1.22.0 头文件） |
| `facehi_ort122.dll` | Windows 私有 ORT 1.22.0，给自定义算子内部子模型用 |
| `Linux 使用说明.docx` / `.txt` | Linux 用法 |
| `Windows 使用说明.docx` / `.txt` | Windows 用法 |
| `样例图.zip` | 仓库 `data/` 17 张样例 |
| `facehi_onnx.py` | 一行调用封装 |
| `requirements.txt` | 钉死 `onnxruntime==1.22.0` |
| `COMPARE.md` | Linux/Windows 环境版本清单 + 与旧 `lib/` 的质量/速度实测 |

ONNX 与对应平台的 `.so` / `.dll` **必须配合使用**。环境版本、与旧 `lib/` 的质量/速度对照见 `COMPARE.md`。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
unzip 样例图.zip
.venv/bin/python -c "from facehi_onnx import remove_highlight; remove_highlight('样例图/1.png', save_to='out.png')"
```

Windows 把 `.venv/bin/` 换成 `.venv\Scripts\`，并使用 `facehi_custom_ops.dll`。
