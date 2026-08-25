# facehi_daily.onnx —— 日常/平衡档

独立的单文件去高光模型（**日常版**）：去高光够用，不像常用档（强力向）那么狠，
也不像「保护细节」取向那么弱。适合日常批量处理证件照时的默认选择。

## 基本信息

| 项目 | 值 |
| --- | --- |
| 文件 | `git-high2/models/facehi_daily.onnx` |
| 大小 | 6,280,130 字节（6.28 MB） |
| SHA-256 | `9d9c7b302384fe47c13714f3c6d756dff26a80f75b9db179099099579b013ee9` |
| 烘焙 mode | `高保真模式`（节点属性 `mode` 与 YAML `default_mode` 均为该值） |
| 档位含义 | 检测=正常、修复=正常、method=混合、process_scale=compromise |
| 内核 | 与 `facehi.onnx` 完全相同的 `ai.facehi:HighlightRemoval` 自定义算子，共用 `lib/libfacehi_custom_ops.so`（Windows 为 `lib/facehi_custom_ops.dll`） |
| 生成方式 | `python git-high2/onnx/make_facehi_onnx.py --preset daily`（完整仓库内运行） |

「高保真模式」是现有 CLI 三档（常用 / 高保真 / 最高质量）里的中间档：
高光检测与修复强度都取 `configs/default.yaml` 的「正常」基准
（修复方法 `mode: 混合`），处理尺度与常用模式一样用 `compromise` 折中档。
与 `facehi.onnx` 相比，只有烘焙进节点属性的 `mode` / `default_mode` 不同，
内嵌的人脸检测/关键点子模型、Haar 级联与三种模式的合并配置表逐字节一致。

> 提示：在当前预设文件下，「常用模式」的修复参数（强力预设）恰好与
> `default.yaml` 基准值相同，检测（灵敏预设）与「正常」只差
> `brow_region_max_fraction`（0.32 vs 0.28）一项上限，因此两档在多数样张上
> 输出一致，仅当眉区高光占比落在该区间时才会分化。日常档的意义在于
> **语义上锁定平衡取向**：后续任何一侧预设调整（加强常用档或收敛正常基准）
> 都会自动体现为两个独立 onnx 的差异，互不影响。

## 用法

```python
import sys; sys.path.insert(0, "git-high2")   # 独立目录内运行则不需要
from facehi_onnx import remove_highlight

remove_highlight("photo.png", variant="daily", save_to="photo_out.png")
```

底层等价于：

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.register_custom_ops_library("git-high2/lib/libfacehi_custom_ops.so")  # 必须
sess = ort.InferenceSession("git-high2/models/facehi_daily.onnx", so)
result, mask = sess.run(None, {"image": bgr_uint8_hwc})   # uint8 BGR [H,W,3]
```

前端 `app_git_high2.py` 左侧「ONNX 模型档位」选
「facehi_daily.onnx · 日常/平衡」即可；对比页建议与 Python 版「高保真模式」配对。

## 验证记录（本分支实测，Ubuntu 24.04 x86-64，onnxruntime 1.29.0 + 仓库自带 so）

1. **全量跑通**：`data/` 全部 17 张 png 走
   `lib/libfacehi_custom_ops.so` + `facehi_daily.onnx` 完整链路（内置人脸检测），
   全部成功。热身后单张 0.066–0.428 秒（648×486 至 1448×1086），
   硬掩码覆盖 1.25%–2.65%，相对原图改动像素 2.57%–4.88%。
2. **与 Python 参考对齐**：对 `data/1、3、7、12、15.png` 与
   `cli_process.load_cli_config("高保真模式")` + `process_image` 对拍：
   MAE ≤ 0.0032、最大像素差 ≤ 10/255、差异像素占比 ≤ 0.504%、
   PSNR ≥ 71.58 dB、硬掩码 IoU ≥ 0.9966——差异量级与 facehi.onnx↔常用模式的
   既有基准相同，仅来自两套人脸推理引擎的亚像素关键点噪声。
3. **烘焙 mode 生效性**：构造探针模型验证内核确实读取节点属性
   `mode="高保真模式"`——只改内嵌 YAML 中「高保真模式」段的修复参数时输出
   显著变化（max diff 59/255），只改「常用模式」段时输出逐位不变。
4. **与 facehi.onnx 的关系**：两模型内嵌子模型/配置表逐字节一致，
   仅 `mode`/`default_mode`/doc_string 不同；`--preset standard` 生成的
   模型与仓库已提交的 `facehi.onnx` 在 mode 与 config_yaml 上逐字节一致。
