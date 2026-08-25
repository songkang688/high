# QA 报告：日常（daily）ONNX 变体

- 审计执行：QA+FIX 云端子代理（slug `claude-fable-5-thinking-xhigh`），分支 `cursor/qa-onnx-daily-6544`（基于 `cursor/three-onnx-git-high1-6544` @ 8066a05）。
- 审计范围：仅 日常 路径（`configs/onnx_daily.yaml` + `models/high_removal_daily.onnx` 及其在 CLI / Studio / 打包脚本中的接线）。未触碰 strong / detail 的 yaml 与 onnx。
- **结论：PASS，未发现真实 bug，无漂移，无需修复**（yaml 未从 default 漂移，onnx 无需重导出）。

## 测试环境

Python 3.12.3 · onnxruntime 1.29.0 · opencv-contrib-python 4.14.0 · mediapipe 1.0.1 ·
精确内核 `cpp/build/libhigh_removal_pyops.so`（本机 g++ 13.3 + ONNX Runtime 1.22.0 头文件全新构建）——
与 `tools/metrics_daily.md` 记载的原始测试环境完全一致。

被测工件（sha256）：

| 文件 | sha256 |
|------|--------|
| `configs/onnx_daily.yaml` | `5428c15184ce85be5c2c4bd48baccd58a54710754e1fa46984596ce425ff9194` |
| `configs/default.yaml` | `7e547cfc266a38db0b757332521c5a1d3f00b75204d13841088220e3b258a0f0` |
| `models/high_removal_daily.onnx` | `5d77ee583523d71e3b928df8544cb2aaf30401a68ee2c24175d74cf97f0555ac` |
| `models/high_removal.onnx` | `ba0b41bd38d4d290d5d8ad2f34232e76805c5be9033bf39864c0f22e4cf54a82` |

## 必测项结果（6/6 PASS）

### 1. onnx_daily.yaml 正文（忽略注释）≡ default.yaml —— PASS

四种方式独立验证，全部为真：

- `yaml.safe_load` 解析后的结构逐项相等；
- 去掉注释与空行后逐行相等；
- 去掉顶部「日常版」注释块后的正文与 `default.yaml` **逐字节**相等（注释头保留，以 `# 日常版` 开头）；
- 内嵌校验：`high_removal_daily.onnx` 的 `config_yaml` 属性 == `configs/onnx_daily.yaml` 文件字节；
  `high_removal.onnx` 的 `config_yaml` 属性 == `configs/default.yaml` 文件字节。

### 2. session.run(daily onnx, 精确内核) ≡ process_image(default.yaml) ≡ process_image(onnx_daily.yaml) —— PASS

`data/{1,15,16}.png` 三图，result 与 hard_mask 均 `np.array_equal`（配置构建方式与
`highlight_removal.exact_kernel._build_config` 完全一致：yaml + 默认 CPU 运行时 + 关闭可视化）：

| 图片 | sess ≟ process_image(onnx_daily.yaml) | sess ≟ process_image(default.yaml) | 硬掩码 px | 修改像素 px |
|------|------|------|------|------|
| 1.png | ✅ | ✅ | 21336 | 43866 |
| 15.png | ✅ | ✅ | 43924 | 80917 |
| 16.png | ✅ | ✅ | 33500 | 69531 |

硬掩码像素数（21336 / 43924 / 33500）与已提交的 `tools/metrics_daily.md` 完全一致，交叉印证环境复现无偏差。

### 3. daily 输出 ≡ 现有 high_removal.onnx —— PASS

同一精确内核下 `session.run(high_removal_daily.onnx)` 与 `session.run(high_removal.onnx)`
在 1.png 上 result 与 hard_mask 均逐位相同（额外在 15.png、16.png 上也验证为逐位相同，超出必测范围）。

### 4. run_high_onnx.py `--preset daily` → high_removal_daily.onnx —— PASS

- 静态：`PRESET_MODELS["daily"] == ROOT / "models" / "high_removal_daily.onnx"`；
- 动态端到端：`python tools/run_high_onnx.py --preset daily -i data/1.png -o out.png`
  实际运行成功，输出 PNG 解码后与直接 `session.run(high_removal_daily.onnx)` 的 result **逐位相同**。

### 5. Studio 默认预设为 日常 —— PASS

`app_studio.py`：`PRESET_DAILY = "日常"`，`DEFAULT_PRESET = PRESET_DAILY`，
预设 Radio `value=DEFAULT_PRESET`，且 `PRESETS["日常"] = (models/high_removal_daily.onnx, configs/onnx_daily.yaml)`。无需改动。

### 6. 打包脚本包含 daily onnx —— PASS

`tools/export_git_high1.sh` 的 cp 列表已包含 `models/high_removal_daily.onnx`（与 strong / detail / 默认模型并列），无需加行。

## Bug 猎捕记录（除必测项外额外检查，均未发现问题）

- **Studio 配置缓存污染**：`studio_config()` 返回共享缓存 dict 且未 deepcopy 即传入
  `process_image` —— 检查确认 `process_image` 首行即 `config = deepcopy(config)`
  （`highlight_removal/pipeline.py`），共享缓存不可能被流水线改写，非 bug。
- **Studio 预设切换会话缓存**：`onnx_session.get_session` 以 `(model, ops)` 为 key，
  路径变化时重建会话——日常↔其他预设切换正确，不会串模型。
- **可复现性**：重跑已提交的 `python tools/metrics_daily.py`，重新生成的
  `tools/metrics_daily.md` 与提交版本**逐字节一致**（git diff 为空）；
  重跑 `python tools/three_onnx_audit.py`，独立审计结论 PASS（其中日常相关断言：
  内嵌配置一致、session≡process_image、日常≡default 流水线（3 图）、daily onnx≡high_removal.onnx 均为 True）。
- **文档接线**：`git-high1/README.md`、`run_high_onnx.py` 文档字符串、`app_studio.py`
  预设说明中关于日常档「= default.yaml 平衡效果、与 high_removal.onnx 逐位相同」的声明与实测一致，无误导。

## 复现方法

```bash
# 环境：pip install onnxruntime onnx "opencv-contrib-python>=4.8,<5" "opencv-python>=4.8,<5" mediapipe pyyaml
# 内核：cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
#           -DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0 && make -C cpp/build -j$(nproc)
python tools/metrics_daily.py       # 日常版指标 + 一致性（应与 tools/metrics_daily.md 逐字节一致）
python tools/three_onnx_audit.py    # 三档独立审计（应 PASS）
python tools/run_high_onnx.py --preset daily -i data/1.png -o /tmp/out.png   # CLI 端到端
```
