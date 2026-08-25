# 三档 ONNX（强力 / 日常 / 保护细节）综合审核报告

**结论：PASS（三项复核全部通过）**

- 审核分支：`cursor/onnx-audit-ce3b`（基于基线 `origin/cursor/git-high2-win-dll-14b9` @ `5f9eb23`，未 merge main）
- 审核范围：仅 facehi 谱系（`ai.facehi:HighlightRemoval` 单文件模型 + 自定义算子库）
- 审核对象（三个变体分支最新提交）：
  - 强力档 `origin/cursor/onnx-strong-5d85` @ `df6b688`（PR #12）
  - 日常档 `origin/cursor/onnx-daily-cccf` @ `3a6cb01`（PR #14）
  - 保护细节档 `origin/cursor/onnx-detail-5e39` @ `fc3bda1`（PR #13）
- 独立复核环境：Linux x86_64、Python 3.12.3、onnxruntime 1.29.0、OpenCV 4.14.0、numpy 2.4.4（CPUExecutionProvider）
- 本次为对父代理结论的**独立复核**：所有 blob、merge-tree、推理数据均在本环境重新计算，未复用父代理产物。

---

## 复核一：三对 merge-tree 必须 CLEAN —— 通过

### 1a. 四个共享文件 blob 三支逐一比对（`git rev-parse <branch>:<path>`）

| 共享文件 | strong-5d85 | daily-cccf | detail-5e39 | 一致 |
| --- | --- | --- | --- | --- |
| `git-high2/facehi_onnx.py` | `e133754` | `e133754` | `e133754` | 是 |
| `git-high2/app_git_high2.py` | `c7fd320` | `c7fd320` | `c7fd320` | 是 |
| `onnx/make_facehi_onnx.py` | `860be4d` | `860be4d` | `860be4d` | 是 |
| `git-high2/README.md` | `68750e6` | `68750e6` | `68750e6` | 是 |

完整 blob：`e133754c8026eab0452854f4f82bf65bce767af8`、`c7fd320933860742914727df668b38f1aea0c169`、`860be4d84d995e579d29d1d21c4ffcc62f83ad4a`、`68750e6497d0ef3f281ebb81872cf21ff91ce3db`。四个共享文件三支**完全相同**，与父代理结论一致。

### 1b. 三对 `git merge-tree --write-tree`（git 2.43.0，自动取 merge-base）

| 分支对 | exit code | 输出 | 结论 |
| --- | --- | --- | --- |
| strong-5d85 × daily-cccf | 0 | 仅树 OID `0353624`，1 行 | CLEAN |
| strong-5d85 × detail-5e39 | 0 | 仅树 OID `782e563`，1 行 | CLEAN |
| daily-cccf × detail-5e39 | 0 | 仅树 OID `ff4ce7c`，1 行 | CLEAN |

三对输出均只有一行合并树 OID，**无任何 CONFLICT 段**（exit 0；有冲突时 merge-tree 会返回 1 并列出冲突文件）。三支两两可干净合并。

---

## 复核二：三 onnx + 基线 so，data/*.png 17 张全量推理 —— 通过

### 方法

- 从各分支 blob 直接提取（不依赖工作树）：三个变体 onnx + 基线 `facehi.onnx`（blob `16a9b87`，四支同一文件）+ 基线 `libfacehi_custom_ops.so`（blob `5e55957`，四支同一文件，即基线分支已验证的那份）。
- `facehi_daily.onnx` SHA-256 = `61ad397c1b07f92a657dc2ab16aa75d328fc26849928e58ead71f047b386ad67`，与该分支 `DAILY.md` 记录逐字节一致。
- 四档共用同一份共享 `facehi_onnx.py`（blob `e133754`）加载，`FacehiOnnx(variant=...)`，对 `data/1.png`–`17.png` 全量推理。
- 强度指标：输出 vs 输入的**全图平均绝对差**（float64，BGR 三通道）。

### 逐图结果

| 图 | 强力 mean | 日常 mean | 细节 mean | 强>日>细 | 日常 vs facehi.onnx maxabs | ≥2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.png | 0.2010 | 0.1038 | 0.0502 | Y | 29 | Y |
| 2.png | 0.3403 | 0.2093 | 0.0886 | Y | 22 | Y |
| 3.png | 0.2235 | 0.1431 | 0.0610 | Y | 19 | Y |
| 4.png | 0.2219 | 0.0921 | 0.0503 | Y | 34 | Y |
| 5.png | 0.1594 | 0.0622 | 0.0361 | Y | 32 | Y |
| 6.png | 0.2356 | 0.0926 | 0.0476 | Y | 33 | Y |
| 7.png | 0.1717 | 0.1123 | 0.0521 | Y | 22 | Y |
| 8.png | 0.1256 | 0.0636 | 0.0334 | Y | 23 | Y |
| 9.png | 0.1976 | 0.0794 | 0.0444 | Y | 32 | Y |
| 10.png | 0.2612 | 0.1026 | 0.0541 | Y | 33 | Y |
| 11.png | 0.1968 | 0.0742 | 0.0427 | Y | 27 | Y |
| 12.png | 0.2796 | 0.1765 | 0.0776 | Y | 23 | Y |
| 13.png | 0.2008 | 0.1122 | 0.0498 | Y | 27 | Y |
| 14.png | 0.1196 | 0.0615 | 0.0333 | Y | 22 | Y |
| 15.png | 0.2228 | 0.1169 | 0.0580 | Y | 21 | Y |
| 16.png | 0.3368 | 0.2051 | 0.0858 | Y | 23 | Y |
| 17.png | 0.1817 | 0.0859 | 0.0513 | Y | 25 | Y |

### 判定

| 验收项 | 门槛 | 实测 | 结论 |
| --- | --- | --- | --- |
| 强力 > 日常 > 细节（严格排序） | ≥12/17 | **17/17** | PASS |
| 日常 vs `facehi.onnx` 输出 maxabs ≥ 2 | ≥15/17 | **17/17**（最小 19，最大 34） | PASS |

17 张图上排序无一例外；日常档与默认 `facehi.onnx` 输出的最大像素差 19–34，远超「≥2」门槛，证明日常档是真实独立档位而非默认档的重打包。

---

## 复核三：同一加载器 variant=strong/daily/detail 定位 —— 通过

把三个变体 onnx 与基线 `facehi.onnx` 拷入同一 `models/` 目录（临时目录 `/tmp/audit/git-high2/models/`），用同一份共享 `facehi_onnx.py`：

| variant | `find_model` 返回 | 文件名匹配 `MODEL_VARIANTS` | `FacehiOnnx(variant=...)` 建会话 |
| --- | --- | --- | --- |
| `default` | `models/facehi.onnx` | 是 | 成功 |
| `strong` | `models/facehi_strong.onnx` | 是 | 成功 |
| `daily` | `models/facehi_daily.onnx` | 是 | 成功 |
| `detail` | `models/facehi_detail.onnx` | 是 | 成功 |

- 中文别名（`强力`/`日常`/`平衡`/`细节`/`保护细节`/`默认`/`常用`）经 `normalize_variant` 全部归一到与标准键相同的文件——逐一验证一致。
- 四档会话共用同一 `lib/libfacehi_custom_ops.so`（基线 blob `5e55957`）注册成功，复核二的 68 次推理（17 图 × 4 档）即在这些会话上完成，全部正常出图。

---

## 总结

| # | 复核项 | 结论 |
| --- | --- | --- |
| 1 | 三对 merge-tree CLEAN + 四共享文件 blob 三支一致 | PASS |
| 2a | 强力>日常>细节 ≥12/17 | PASS（17/17） |
| 2b | 日常 vs facehi.onnx maxabs≥2 ≥15/17 | PASS（17/17，min=19） |
| 3 | 同一加载器 variant=strong/daily/detail 均可定位并运行 | PASS |

**最终结论：PASS。** 独立复核与父代理结论一致：三个变体分支共享文件零分歧、两两可干净合并；三档效果强度严格分层；日常档与默认档真实可区分；三档模型可在同一 `models/` 目录下由同一加载器按 variant 正确选择。
