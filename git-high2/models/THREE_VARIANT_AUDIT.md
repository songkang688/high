# 三档 ONNX 综合审核报告（第三轮，覆盖第二轮）

**结论：FAIL（仅判据 1·两两 merge 仍全冲突；判据 2 数值回归与判据 3 运行时统一
接口全部 PASS）。**
三个模型本体与统一 `MODEL_VARIANTS` 加载语义均已达标，无需重新导出任何 onnx、
无需改任何运行时行为；FAIL 纯粹是文本层问题：三分支按第二轮建议各自实现了
同一套四档规范，但**同一区域三种写法**（辅助函数名 / 缓存结构名 / 注释与 README
措辞互不相同），两两合并在四个共享文件上依然全部冲突。应 resume 的档位与
逐文件处方见「五」，处方已在本轮用模拟提交验证三对合并全清。

## 一、审核对象（三分支最新提交，facehi 谱系）

| 档位 | 分支 / PR | 审核 commit | 模型 | sha256(前16) |
| --- | --- | --- | --- | --- |
| 基线 | cursor/git-high2-win-dll-14b9 | `5f9eb23` | facehi.onnx | `65fc13a2317d2412` |
| 强力 | cursor/onnx-strong-5d85 · PR #12 | `c7cb46c`（四档 registry） | facehi_strong.onnx | `2d89a37f2dddac18` |
| 日常 | cursor/onnx-daily-cccf · PR #14 | `3a6cb01`（对齐 + 独立日常 onnx） | facehi_daily.onnx | `61ad397c1b07f92a` |
| 细节 | cursor/onnx-detail-5e39 · PR #13 | `7871c6d`（纯文件名四档；onnx 未重导出，与第二轮同一文件） | facehi_detail.onnx | `86429520e9ad7a68` |

四分支 `git-high2/lib/libfacehi_custom_ops.so` 为同一 blob（`5e55957a…`），
本轮全部推理共用基线分支工作区内的该 so。模型行为与第二轮完全一致
（本轮 Ldrop 采用 L\* 0–100 标度，数值 = 第二轮 L 0–255 标度 ÷ 2.55，
17/17 张逐项吻合），证实三次「对齐」提交确实未动模型文件。

环境：Ubuntu x86-64，onnxruntime 1.29.0（CPU EP），OpenCV-headless 5.0，numpy 2.x。

## 二、判据 1：两两 merge-tree —— FAIL

按 `git merge-tree $(git merge-base A B) A B`（旧式）与
`git merge-tree --write-tree A B`（ort，真实 merge 同引擎）双口径实测，
结论一致：**三对两两合并在四个共享文件上全部冲突**。三对 merge-base 均为
基线 `5f9eb23`。仓库根 `README.md` 四分支同一 blob 无冲突；冲突的 README
是 `git-high2/README.md`。三个 onnx 模型文件名互不相同，零冲突。

| 冲突块数（旧式 merge-tree） | 强力×日常 | 强力×细节 | 日常×细节 |
| --- | --- | --- | --- |
| git-high2/facehi_onnx.py | 10 | 6 | 10 |
| git-high2/app_git_high2.py | 12 | 3 | 12 |
| onnx/make_facehi_onnx.py | 7 | 4 | 5 |
| git-high2/README.md | 4 | 4 | 5 |

根因：第二轮要求的**语义**统一已完成（同一 `MODEL_VARIANTS: dict[str,str]`
四键字典、`variant=None→"default"`、分档环境变量
`FACEHI_ONNX_MODEL{,_STRONG,_DAILY,_DETAIL}`、缺文件返 None、四档下拉、
单一仓库根生成脚本——三支全部做到），但三支在**同一行区间**写了三套文本：

| 文件 | 强力 c7cb46c | 日常 3a6cb01 | 细节 7871c6d |
| --- | --- | --- | --- |
| facehi_onnx.py 环境变量推导 | `_MODEL_ENV_VARS` 字典 | `_variant_env_name()` 函数 + `VARIANT_ALIASES`/`normalize_variant`（中文别名） | `_variant_env_key()` 函数 |
| app 会话缓存 | `_ONNX_CACHE`（dict 嵌套） | `_ONNX_WRAPPERS` + `_ONNX_LOAD_SECONDS`（平面双 dict） | `_ONNX_CACHE`（dict 嵌套） |
| app 下拉选项 | `_ONNX_VARIANT_LABELS` 列表 + 循环拼 choices（缺档标「未提供」） | `ONNX_VARIANT_CHOICES` dict（label→variant） | `ONNX_VARIANT_BY_LABEL` dict + `list()` |
| make 脚本 PRESETS 键 | 仅 `strong`（预留槽位注释） | `standard/strong/daily/detail` 四档齐全 | `strong/detail`（修复预设键名用 `removal_preset`） |
| git-high2/README.md | 档位表（strong 视角措辞） | 四档全景措辞 | 双模型措辞 |

典型冲突块（facehi_onnx.py 同一函数体三种写法）：强力
`env_name = _MODEL_ENV_VARS.get(key)` ↔ 日常 `key = normalize_variant(variant)`
↔ 细节 `key = variant or "default"`；app 同一缓存读写点
`_ONNX_CACHE.get(variant)` ↔ `_ONNX_WRAPPERS.get(variant)`；make 脚本
PRESETS 字典头部注释与档位条目三种排布；README 同一行
`models/facehi.onnx` 三种注释文案。

## 三、判据 2：17 张全量数值回归 —— PASS

三档模型 `git show` 提取至独立目录 + 基线 so，`data/*.png` 17 张
4 模型 × 17 = 68 次 `session.Run` 全部成功。共同 mask = 三档
`highlight_mask>0` 并集；Ldrop = mask 内 mean(L\*原 − L\*出)（L\* 0–100）；
maxabs = 全图逐像素任意通道最大绝对差。

| 门槛 | 要求 | 实测 | 判定 |
| --- | --- | --- | --- |
| 强度阶梯 强力>日常>细节 | ≥12/17 逐张严格递减 | **17/17**，均值 3.970 > 1.968 > 0.911 | **PASS** |
| 日常 vs 基线 facehi.onnx | ≥15/17 张 maxabs≥2 | **17/17**（最小 19，样本 3） | **PASS** |

第二轮警示的「日常档 = 基线换标签」已彻底修复：日常档现为独立「日常模式」
中值配置，与基线逐张 maxabs 19–34，且稳定位于强力（vs 强力 maxabs 28–55）
与细节（vs 细节 maxabs 19–41）之间。

| 图 | 并集px | Ldrop强力 | Ldrop日常 | Ldrop细节 | 递减 | 日常↔基线 maxabs | 日常↔强力 | 日常↔细节 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 21328 | 3.719 | 1.909 | 0.863 | ✓ | 29 | 37 | 29 |
| 2 | 25948 | 5.209 | 3.222 | 1.311 | ✓ | 22 | 32 | 37 |
| 3 | 4960 | 4.230 | 2.676 | 1.097 | ✓ | 19 | 29 | 41 |
| 4 | 21192 | 4.060 | 1.650 | 0.847 | ✓ | 34 | 47 | 26 |
| 5 | 15356 | 4.162 | 1.530 | 0.837 | ✓ | 32 | 43 | 26 |
| 6 | 4992 | 4.814 | 1.758 | 0.794 | ✓ | 33 | 43 | 19 |
| 7 | 25196 | 2.498 | 1.611 | 0.675 | ✓ | 22 | 30 | 34 |
| 8 | 15528 | 3.040 | 1.516 | 0.745 | ✓ | 23 | 30 | 34 |
| 9 | 19796 | 3.783 | 1.514 | 0.815 | ✓ | 32 | 43 | 24 |
| 10 | 20696 | 5.116 | 1.940 | 0.968 | ✓ | 33 | 55 | 26 |
| 11 | 18196 | 4.389 | 1.541 | 0.857 | ✓ | 27 | 50 | 21 |
| 12 | 23244 | 4.680 | 2.951 | 1.263 | ✓ | 23 | 30 | 38 |
| 13 | 19824 | 3.875 | 2.164 | 0.904 | ✓ | 27 | 34 | 28 |
| 14 | 19836 | 2.889 | 1.469 | 0.754 | ✓ | 22 | 30 | 33 |
| 15 | 43936 | 2.571 | 1.370 | 0.656 | ✓ | 21 | 28 | 33 |
| 16 | 33484 | 5.100 | 3.132 | 1.259 | ✓ | 23 | 32 | 37 |
| 17 | 22228 | 3.355 | 1.506 | 0.849 | ✓ | 25 | 43 | 23 |
| **均值** | 20925 | **3.970** | **1.968** | **0.911** | 17/17 | min 19 | min 28 | min 19 |

## 四、判据 3：合并形态 variant 四键解析 —— PASS

- **三分支各自加载器**（模型文件齐备时）：`find_model` 对
  `default/strong/daily/detail` 四键解析结果三支完全一致
  （facehi.onnx / facehi_strong.onnx / facehi_daily.onnx / facehi_detail.onnx），
  未知键均返 None——语义已收敛，冲突确为纯文本。
- **临时合并目录**（日常档加载器 + 四个模型文件齐备）：
  `remove_highlight(img, variant=k)` 四键实跑全部成功且输出两两分化
  （default/strong/daily/detail 改动像素 29508/29225/21527/16339）；
  中文别名 `强力/日常/保护细节/常用` 归一正确。
- **缺文件行为**：移走 facehi_detail.onnx 后 `find_model("detail")` 返 None
  （不抛异常），符合要求。

## 五、若 FAIL 应 resume 的档位（处方已模拟验证）

**resume 强力档（cursor/onnx-strong-5d85，PR #12）与细节档
（cursor/onnx-detail-5e39，PR #13）；日常档（cursor/onnx-daily-cccf，PR #14）
无需 resume。** 三支文本两两互异，至少两支必须改；统一基准取**日常档
`3a6cb01` 的四个共享文件文本**，理由：

1. 它是唯一一份**四档完备**的文本：make 脚本 PRESETS 含
   standard/strong/daily/detail 全部四档——其中 `strong` 预设 overrides 与
   强力分支自己的逐键相同（含 d4cd8cb 加强后的 `extreme_core_extra_l: 0`、
   `final_blend_alpha: 1.0`），`detail` 预设与细节分支语义一致
   （仅 `intensity_preset`/`removal_preset` 键名之差）；README 四档全景；
   加载器为功能超集（多出中文别名，环境变量约定与另两支相同）。
2. 强力文本缺 daily/detail 预设、细节文本缺 daily 预设：若以它们为基准，
   另两支需在同一锚点各自增补条目，两两之间仍会冲突。

处方（每支一条命令级操作，别的什么都不要改）：

```bash
# 在 cursor/onnx-strong-5d85 与 cursor/onnx-detail-5e39 上分别执行：
git checkout origin/cursor/onnx-daily-cccf -- \
  git-high2/facehi_onnx.py git-high2/app_git_high2.py \
  onnx/make_facehi_onnx.py git-high2/README.md
git commit -am "对齐日常档统一文本（四共享文件字节级一致）"
# 各自的 models/facehi_*.onnx 与 models/STRONG.md / DETAIL.md 保持不动。
```

本轮已按上述处方构造模拟提交（sim-strong `6988aa3e`、sim-detail `1958165`，
均未推送）实测：三对两两 merge-tree（旧式 + ort 双口径）**全部 CLEAN，
0 冲突标记**。执行后共享文件三支同 blob、分支独有文件互不相交，合并顺序任意。

若还想保留强力档「缺档标注未提供」的下拉增强，必须三支同字节改动，
建议放弃，留给三档合流后的整合 PR 单独做。

## 六、复现

```bash
git show origin/cursor/onnx-strong-5d85:git-high2/models/facehi_strong.onnx > /tmp/audit/models/facehi_strong.onnx
git show origin/cursor/onnx-daily-cccf:git-high2/models/facehi_daily.onnx  > /tmp/audit/models/facehi_daily.onnx
git show origin/cursor/onnx-detail-5e39:git-high2/models/facehi_detail.onnx > /tmp/audit/models/facehi_detail.onnx
# 四模型同用基线 git-high2/lib/libfacehi_custom_ops.so 注册后
# sess.run(["result","highlight_mask"], {"image": bgr_uint8_hwc}) 跑 data/*.png 17 张。
# merge 检查（三对、双口径）：
#   git merge-tree $(git merge-base A B) A B | grep -c '<<<<<<<'
#   git merge-tree --write-tree --name-only A B
# 口径：共同 mask = 三档 highlight_mask 并集；Ldrop 用 L*（0–100，
# = OpenCV Lab L×100/255）；maxabs = 全图逐像素任意通道最大绝对差。
```
