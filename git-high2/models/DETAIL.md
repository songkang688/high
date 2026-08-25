# facehi_detail.onnx —— 保护细节版

`facehi_detail.onnx`（约 6.27MB）是 `facehi.onnx` 的**保护细节**变体：
五官/皮肤纹理尽量保留，去高光故意弱一点。适合轻中度油光、要求皮肤质感
（证件照要"像本人"而非磨皮感）的场景；重油光、追求残留最少时仍用默认的
`facehi.onnx`（常用模式）。

两个模型**共用同一个自定义算子库**（`lib/libfacehi_custom_ops.so` /
`facehi_custom_ops.dll`）和同一个 C++ 内核（`ai.facehi:HighlightRemoval`
自定义算子）——区别只在烘焙进节点属性的配置：`mode="保护细节"` +
内嵌单模式 YAML。没有第二套内核。

## 烘焙配置（与常用模式的差异）

生成命令：`python onnx/make_facehi_onnx.py --preset detail`（仓库内）。
组合 = 检测「弱」预设 + 修复「削弱」预设 + method 混合 + 纹理优先键 +
保护区加宽，全部来自仓库现有预设文件与 C++ 内核已支持的键：

| 环节 | 键 | 常用模式 | 保护细节 | 作用 |
| --- | --- | --- | --- | --- |
| 检测 | 整段 ← `detection_sensitivity_presets.yaml` | 灵敏 | **弱** | 阈值抬高（`lab_l_threshold` 178→198 等）、掩码不膨胀反腐蚀（dilate 2→0 / erode 0→1）、区域面积上限收紧 → 掩码更小更贴高光核心 |
| 修复 | 整段 ← `removal_intensity_presets.yaml` | 强力 | **削弱** | 压制强度 0.94→0.68、最终混合 0.97→0.82、inpaint 半径 6→3、色度回复 0.38→0.18 → 改动更轻，残留可略多 |
| 修复 | `mode` | 混合 | **混合**（削弱预设原为保真，此处改回混合） | 近饱和白斑仍有 Telea 兜底 |
| 修复 | `extreme_core_extra_l` | 10 | **30** | Telea 强修复核心阈值抬到 `lab_l_threshold+30`（弱检测下 198+30=228，只处理近饱和白斑），其余油光全走保真 Lab 压制，避免 Telea 大面积平滑抹掉皮肤纹理（参考 `configs/experimental_better.yaml` 的纹理优先思路，只用现有内核支持的键） |
| 修复 | `texture_preserve_strength` | 0.70 | **1.0** | 保真压制的高频纹理回加拉满（内核系数 0.16×strength 封顶） |
| 修复 | `edge_protect_strength` | 0.38 | **0.60** | 边缘保护再抬一档，纹理边界少动 |
| 分区 | `protect_expand_radius` | 5 | **7** | 眼鼻嘴保护区外扩更宽 |
| 分区 | `eye_protect_extra_radius` | 1.0 | **1.6** | 眼周额外保护半径加大 |
| 底座 | `pipeline` | compromise 尺度 + ROI 等四固定键 | 相同 | 与常用/高保真同底座 |

完整合并配置内嵌在模型节点属性 `config_yaml` 中（`modes: 保护细节`），
可用 `onnx.load` 读取核对。

## 使用

```python
import sys; sys.path.insert(0, "git-high2")   # 独立目录内运行则不需要
from facehi_onnx import remove_highlight

out = remove_highlight("photo.png", variant="detail")            # 保护细节版
out = remove_highlight("photo.png")                              # 常用（默认）
```

底层等价于：

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.register_custom_ops_library("git-high2/lib/libfacehi_custom_ops.so")  # 同一个库
sess = ort.InferenceSession("git-high2/models/facehi_detail.onnx", so)
result, mask = sess.run(None, {"image": bgr_uint8_hwc})
```

前端（`app_git_high2.py`）ONNX 版页签内可直接切换「常用 / 保护细节」模型。
环境变量 `FACEHI_ONNX_MODEL_DETAIL` 可覆盖保护细节模型路径
（常用模型仍用 `FACEHI_ONNX_MODEL`）。

## 验证记录（本交付包实测）

Ubuntu x86-64、`lib/libfacehi_custom_ops.so`（OpenCV 4.14 静态 + IPP）、
onnxruntime 1.29.0，`data/` 全部 17 张 + `cpp/golden` 黄金样本：

- **so 加载 + 17/17 张全量跑通**，另验证注入黄金关键点（`landmarks` 可选输入）通路正常。
- **改动像素更少：17/17 张全部低于常用模式**。改动像素占比均值
  3.662% → 2.528%，对原图 MAE 均值 0.183 → 0.054，硬掩码面积均值
  20922px → 12148px（约 −42%）。
- **眼鼻嘴保护区零误伤、周边更干净**：黄金 protect 掩码**内部**改动像素
  两个模型均为 0（max|Δ|=0，强制回写在全分辨率生效）；掩码外扩 6px 环带内
  的改动像素，样本 1：1848 → 698；样本 13：1498 → 458（约 −62%/−69%）。
- **去高光故意更弱（残留略多，符合定位）**：以常用模型硬掩码为共同参照区，
  处理后区域平均 L 压暗 8.40 → 2.32；在保护细节模型自己检出的核心区内
  压暗均值 3.67（仍有效，但明显弱于常用）。
- **mode 属性生效探针**：构造双模式 config_yaml（常用模式 + 保护细节，
  `default_mode` 故意设为常用模式）只改节点 `mode` 属性——mode=保护细节 时
  输出与本模型**位级一致**（与常用 max diff=55）；mode 置空时回落
  default_mode，与 facehi.onnx 位级一致；mode=不存在的名字时会话创建报错
  并指出模式名。证明 `mode="保护细节"` 被 C++ 内核真实读取并驱动配置选型，
  而非 default_mode 兜底的巧合。

复现口径：改动像素 = 任一通道 |Δ|>0；L 为 OpenCV Lab 的 L 通道（0–255）。
