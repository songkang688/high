# 三档 ONNX 综合审核报告（第二轮，覆盖旧口径）

**结论：FAIL（仅第 6 条·代码集成冲突；第 1–5 条数值标准全部通过）。**
三个模型本体的行为已全部达标，无需重新导出任何 onnx；FAIL 是产品/集成问题：
三分支对 `facehi_onnx.py` / `app_git_high2.py` 的改法两两 merge 全冲突，
任何单分支 checkout 只能选到「默认 + 本档」两档，无法在同一目录同时选三档。
应 resume 的档位与改法见文末。

## 审核对象（三分支最新提交）

| 档位 | 分支 | 审核 commit | 模型 | md5 | 烘焙 mode |
| --- | --- | --- | --- | --- | --- |
| 基线 | cursor/git-high2-win-dll-14b9 | `5f9eb23` | facehi.onnx | `ca48cb2d…` | 常用模式 |
| 强力 | cursor/onnx-strong-5d85 | `425ca03`（重导出在 `d4cd8cb`） | facehi_strong.onnx | `9df2d1c0…` | 强力模式 |
| 日常 | cursor/onnx-daily-cccf | `fbcd37c` | facehi_daily.onnx | `39900275…` | 日常模式 |
| 细节 | cursor/onnx-detail-5e39 | `e1776be`（仅改 DETAIL.md，9+/2−，onnx 与 `8360d81` 相同） | facehi_detail.onnx | `1c7902b2…` | 保护细节 |

谱系核验：四个模型节点域均为 `ai.facehi:HighlightRemoval`，二进制内**均无**
`high_removal` 字样（facehi 谱系纯净，未混入 high_removal_*.onnx 谱系）。

烘焙参数探针（`onnx.load` 读节点属性 `mode` + `config_yaml` 对应模式段）：

| 键 | 常用(基线) | 强力 | 日常 | 保护细节 |
| --- | --- | --- | --- | --- |
| lab_l_threshold | 178 | 178 | 188 | 198 |
| hsv_v_threshold | 178 | 178 | 190 | 202 |
| extreme_core_extra_l | 10 | **0** | 19 | 30 |
| final_blend_alpha | 0.97 | **1.0** | 0.905 | 0.82 |

强力档 `d4cd8cb` 声称的重导出（extra_l 10→0、混合 alpha 拉满）已在模型内证实；
日常档 `fbcd37c` 声称的独立「日常模式」中值配置（lab_l 188 / hsv_v 190 /
blend 0.905）同样证实，不再等于 facehi.onnx（md5 亦不同）。

## 实测方法与口径（本报告为准，覆盖旧报告）

- 环境：Ubuntu x86-64，onnxruntime 1.29.0（CPU EP），OpenCV-headless 5.0，numpy 2.x。
- 用 `git show <branch>:git-high2/models/*.onnx` 提取三档模型；四档共用**基线**的
  `git-high2/lib/libfacehi_custom_ops.so`（44.3MB，四模型均正常注册加载）。
- 样本：`data/*.png` 全量 17 张；4 模型 × 17 张 = 68 次 `session.Run`。
- **共同 mask 口径 = 三档 `highlight_mask` 的并集（union）**；同时给交集
  （intersection）作对照，两口径结论一致。L 为 OpenCV Lab 的 L 通道（0–255），
  Ldrop = mask 内 mean(L_原) − mean(L_出)。
- 保护区误伤两个口径：① `cpp/golden/{1,13}_protect.png` 黄金保护掩码内部
  改动；② 全 17 张三 mask 并集**之外**像素的 MAE（掩码外不该动的区域）。

## 硬标准逐条判定

| # | 标准 | 结果 | 关键数字 |
| --- | --- | --- | --- |
| 1 | 3×17 跑通且非原图拷贝 | **PASS** | 68/68 次 Run 成功；0 张输出与原图逐位相同 |
| 2 | 强力>日常>细节（共同 mask 内 Ldrop，≥12/17 严格递减且均值递减） | **PASS** | 并集口径 **17/17** 严格递减；均值 **10.12 > 5.02 > 2.32**。交集口径同 17/17，均值 14.45 > 7.73 > 3.67 |
| 3 | 日常 vs 基线 maxabs≥2 至少 15/17 | **PASS** | **17/17**（最小 19，样本 3）；无任何一张逐位相同 |
| 4 | 日常 vs 强力 maxabs≤1 的张数 <10 | **PASS** | **0 张**（最小 maxabs 28，样本 15） |
| 5 | 细节保护区误伤 ≤ 强力 | **PASS** | 黄金 protect 掩码（样本 1：13636px、13：11837px）内四档误伤均为 0（max\|Δ\|=0）；并集外 MAE 细节 **17/17 张 ≤ 强力**（均值 0.0078 vs 0.0166，约 −53%） |
| 6 | 三分支 facehi_onnx.py / app_git_high2.py 改法可共存 | **FAIL** | 两两 `git merge-tree` 全冲突，详见下节 |

## 每张图数据（并集 mask 口径）

| 图 | 并集px | Ldrop强力 | Ldrop日常 | Ldrop细节 | 日常↔基线 maxabs | 日常↔强力 maxabs | 掩码外MAE 强力 | 掩码外MAE 细节 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21328 | 9.48 | 4.87 | 2.20 | 29 | 37 | 0.0153 | 0.0068 |
| 2 | 25948 | 13.28 | 8.22 | 3.34 | 22 | 32 | 0.0195 | 0.0075 |
| 3 | 4960 | 10.79 | 6.82 | 2.80 | 19 | 29 | 0.0227 | 0.0110 |
| 4 | 21192 | 10.35 | 4.21 | 2.16 | 34 | 47 | 0.0181 | 0.0082 |
| 5 | 15356 | 10.61 | 3.90 | 2.13 | 32 | 43 | 0.0118 | 0.0062 |
| 6 | 4992 | 12.28 | 4.48 | 2.02 | 33 | 43 | 0.0183 | 0.0112 |
| 7 | 25196 | 6.37 | 4.11 | 1.72 | 22 | 30 | 0.0217 | 0.0118 |
| 8 | 15528 | 7.75 | 3.87 | 1.90 | 23 | 30 | 0.0112 | 0.0061 |
| 9 | 19796 | 9.65 | 3.86 | 2.08 | 32 | 43 | 0.0145 | 0.0065 |
| 10 | 20696 | 13.05 | 4.95 | 2.47 | 33 | 55 | 0.0162 | 0.0075 |
| 11 | 18196 | 11.19 | 3.93 | 2.19 | 27 | 50 | 0.0143 | 0.0064 |
| 12 | 23244 | 11.93 | 7.52 | 3.22 | 23 | 30 | 0.0170 | 0.0077 |
| 13 | 19824 | 9.88 | 5.52 | 2.31 | 27 | 34 | 0.0169 | 0.0077 |
| 14 | 19836 | 7.37 | 3.75 | 1.92 | 22 | 30 | 0.0109 | 0.0057 |
| 15 | 43936 | 6.55 | 3.49 | 1.67 | 21 | 28 | 0.0184 | 0.0075 |
| 16 | 33484 | 13.01 | 7.99 | 3.21 | 23 | 32 | 0.0199 | 0.0071 |
| 17 | 22228 | 8.56 | 3.84 | 2.17 | 25 | 43 | 0.0157 | 0.0080 |
| **均值** | 20926 | **10.12** | **5.02** | **2.32** | 26.3 | 36.9 | **0.0166** | **0.0078** |

## 第 6 条 FAIL 详情：三分支代码集成冲突

`git merge-tree --write-tree` 实测（任意两两合并均冲突）：

| 合并对 | 冲突文件 |
| --- | --- |
| 强力 × 日常 | README.md、app_git_high2.py、facehi_onnx.py |
| 强力 × 细节 | README.md、app_git_high2.py、facehi_onnx.py、**onnx/make_facehi_onnx.py** |
| 日常 × 细节 | README.md、app_git_high2.py、facehi_onnx.py |

根因是三支各自发明了互不兼容的 `MODEL_VARIANTS` 约定：

| 分支 | MODEL_VARIANTS 结构 | 默认档键 | 未知档位行为 | 会话缓存 |
| --- | --- | --- | --- | --- |
| 强力 | `str → 文件名`（`"default"/"strong"`） | `"default"` | 当文件名兜底（宽松） | `_sessions: dict` |
| 日常 | `None/str → 文件名`（**以 `None` 为默认键**） | `None` | `ValueError`（严格） | `_default_sessions: dict` |
| 细节 | `str → (文件名, 环境变量)` **元组**，新增 `FACEHI_ONNX_MODEL_DETAIL` | `"default"` | `ValueError` | `_sessions`，app 内嵌套 `{"wrappers","load_seconds"}` |

`app_git_high2.py` 同样三套：强力用 `_ONNX_VARIANT_LABELS` 列表 + 按
`find_model` 过滤存在的模型；日常/细节各用一个 `ONNX_VARIANT_CHOICES` dict，
且细节的 `ui_run_onnx` 依赖元组解包 `MODEL_VARIANTS[variant]`。另外日常分支把
生成脚本**复制**到 `git-high2/onnx/make_facehi_onnx.py`（新路径）加 daily 预设，
而强力/细节改的是仓库根 `onnx/make_facehi_onnx.py`（且两支在该文件互相冲突），
合并后会留下两份分叉的生成脚本。

三个 onnx 模型文件本身文件名互不相同、**零冲突**，可直接并存。

## 合并建议（供 resume 执行，本报告不代改）

1. `facehi_onnx.py` 统一为强力分支的 registry 形态并补全四档：
   `MODEL_VARIANTS: dict[str, str] = {"default": "facehi.onnx", "strong":
   "facehi_strong.onnx", "daily": "facehi_daily.onnx", "detail":
   "facehi_detail.onnx"}`。入口把 `variant=None` 归一化为 `"default"`
   （消灭日常分支的 `None` 键）；具名别名未命中时报 `ValueError`，仅当实参以
   `.onnx` 结尾才按文件名兜底；细节分支的按档环境变量能力推广为
   `FACEHI_ONNX_MODEL_<VARIANT大写>`（default 仍用 `FACEHI_ONNX_MODEL`），
   registry 值保持纯文件名字符串、环境变量名按规则推导，不再用元组。
   会话缓存统一 `_sessions: dict[str, FacehiOnnx]`。
2. `app_git_high2.py` 采用强力分支方案：`_ONNX_VARIANT_LABELS` 四行
   （默认/强力/日常/保护细节）+ `find_model(variant) is not None` 过滤缺失模型；
   `_ONNX_CACHE` 用 variant 字符串作平面键（弃用细节分支的嵌套结构）；
   `ui_run_onnx` 不做元组解包。
3. 生成脚本只留仓库根 `onnx/make_facehi_onnx.py` 一份，合并
   `--preset strong / daily / detail` 三个预设；删除日常分支复制出的
   `git-high2/onnx/make_facehi_onnx.py`。
4. README 档位表合并为一张四行表。
5. 三个 onnx 文件按现状直接并存，**不需要重新导出**。

## 若 FAIL 应 resume 的档位

- **优先 resume 日常档（cursor/onnx-daily-cccf）**：它的 `None` 默认键与另两支
  最不兼容，且复制了生成脚本造成双份分叉，需改动最多——按上节 1/2/3 对齐。
- **其次 resume 细节档（cursor/onnx-detail-5e39）**：把 `MODEL_VARIANTS` 值从
  元组改回文件名字符串、环境变量名改为按规则推导、app 缓存去嵌套。
- 强力档代码形态即建议的合并基准，模型与数值三档均达标，无需改模型。

## 复现

```bash
git show origin/cursor/onnx-strong-5d85:git-high2/models/facehi_strong.onnx > /tmp/facehi_strong.onnx
git show origin/cursor/onnx-daily-cccf:git-high2/models/facehi_daily.onnx  > /tmp/facehi_daily.onnx
git show origin/cursor/onnx-detail-5e39:git-high2/models/facehi_detail.onnx > /tmp/facehi_detail.onnx
# 四模型同用基线 git-high2/lib/libfacehi_custom_ops.so 注册后
# sess.run(["result","highlight_mask"], {"image": bgr_uint8_hwc}) 跑 data/*.png 17 张；
# 口径：共同 mask=三档 mask 并集；L=OpenCV Lab L(0-255)；
# 保护区=cpp/golden/{1,13}_protect.png 内部 + 并集 mask 外 MAE。
```
