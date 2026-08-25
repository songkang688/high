# QA 报告：「强力」ONNX 路径（models/high_removal_strong.onnx）

- 结论：**PASS**（未发现需要修复的 bug；强力确实强于日常，无需改参数/重导出）
- 分支：`cursor/qa-onnx-strong-6544`（基于 `cursor/three-onnx-git-high1-6544`，commit `8066a05`）
- 环境：Linux x64，Python 3.12.3，onnxruntime 1.29.0（pip），opencv-contrib-python 4.14.0，
  mediapipe 1.0.1，onnx 1.22.0；`libhigh_removal_pyops.so` 由 `cpp/` 在本机重新构建
  （cmake + g++，`-DONNXRUNTIME_ROOT=/opt/ort/onnxruntime-linux-x64-1.22.0`，仓库内只提交了 Windows DLL）。
- 日期：2026-08-25

## 测试 1：模型存在 + 内嵌配置一致 —— PASS

```bash
python3 - # onnx.load(models/high_removal_strong.onnx) 提取 config_yaml 属性字节
```

- `models/high_removal_strong.onnx` 存在，5 343 797 字节（5.10 MiB），单节点 `ai.high:HighlightRemoval`。
- 内嵌 `config_yaml` 与 `configs/onnx_strong.yaml` **逐字节一致**
  （sha256 均为 `5b2023ebedf72660…d47f1f3e2d`）。
- 附加验证：内嵌 `face_detector_model` / `face_landmarks_model` 与
  `models/face_detector.onnx` / `models/face_landmarks_detector.onnx` 逐字节一致，
  且与 default 模型内嵌的完全相同（三档只差配置，不差人脸网络）。

## 测试 2：精确内核 session.run == process_image(onnx_strong.yaml) —— PASS

data/1.png（1280×960），`libhigh_removal_pyops.so` + `models/high_removal_strong.onnx`
的 `session.run(["result","hard_mask"])`，对比直接调用
`process_image`（配置 = `configs/onnx_strong.yaml` + `exact_kernel._build_config` 同款构建）：

- `result` np.array_equal：**True**
- `hard_mask` np.array_equal：**True**
- 输出确实不等于输入（真的做了去高光）。

## 测试 3：强力 > 日常（两个指标，3/3 图）—— PASS

配置分别取自 `high_removal_strong.onnx` 与 `high_removal_daily.onnx` 的内嵌字节：

```bash
python3 tools/compute_strong_metrics.py --default-model models/high_removal_daily.onnx \
    --strong-model models/high_removal_strong.onnx --images data/1.png data/15.png data/16.png
```

| 图 | mean\|ΔL\|（daily → strong） | 皮肤 L≥200 消减（daily → strong） | 强力更强？ |
| --- | --- | --- | --- |
| 1.png | 8.474 → **10.047** | −9 823 → **−16 983**（17 368→385） | ✔✔ |
| 15.png | 5.969 → **6.767** | −15 616 → **−28 864**（41 563→12 699） | ✔✔ |
| 16.png | 11.436 → **11.659** | −5 446 → **−15 296**（49 396→34 100） | ✔✔ |

两个指标均 3/3 图满足「强力 > 日常」，无需改动 `onnx_strong.yaml` 或重导出。
独立复测 `python3 tools/three_onnx_audit.py` 亦为 **PASS**
（强>日>细 排序 3/3 图、三模型两两不逐位相等、三档 parity 全 True）。

## 测试 4：simple_protect_mode + 无人脸不崩溃 —— PASS

- `configs/onnx_strong.yaml` `regions.simple_protect_mode: true`（内嵌配置同一份，已由测试 1 保证）。
- 无人脸输入：`data/` 17 张全部是人脸照，`assets/app_icon.png` 实测会被 MediaPipe
  检出人脸（图标含人脸图案），仓库内**找不到**真实无人脸照片；改用两张合成图过
  strong 模型 session.run：
  - 640×480 均匀随机噪声：不崩溃，`result` 与输入逐位相同、`hard_mask` 全零；
  - 400×300 纯灰图：不崩溃，同上。
  即 process_image 的"未检出人脸→原样返回+零掩码"路径经 ONNX 会话验证正常。

## 测试 5：Studio / run_high_onnx --preset strong —— PASS，未发现强力独有 bug

实测（非只读代码审查）：

- `python3 tools/run_high_onnx.py --preset strong -i data/1.png -o out.png --mask mask.png`
  正常退出；输出 PNG 与 mask 均与直接 session.run 强力模型**逐位相同**。
- `app_studio.run_python_engine(img, "强力")` 与 `app_studio.run_onnx_engine(img, "强力")`
  输出逐位相同（result 与 hard_mask）；即对比模式在强力档下 MAE 恒为 0 的声明成立。
- 预设隔离：强力跑完后切"日常"，输出确实不同，且日常档 Python/ONNX 依旧逐位相同；
  `studio_config` 的按预设缓存未被污染（`process_image` 入口 deepcopy，实测缓存里
  strong 的 0.99/165、daily 的 0.94 均完好）。
- `app_studio.build_app()` 正常构建（Gradio 页面可创建）。

### 给集成者的备注（共享代码，本分支刻意不改，避免合并冲突）

1. `highlight_removal/onnx_session.py::availability()` 只检查
   `models/high_removal.onnx`（DEFAULT_MODEL_PATH）是否存在。若强力模型单独缺失，
   studio 启动时仍显示"ONNX 引擎就绪"，选强力档运行时才报
   `OnnxEngineUnavailable`（错误进状态栏，不崩溃）。三档共享的行为，非强力独有。
2. 同文件 `get_session` 的缺模型提示统一写"请运行
   `python tools/export_high_removal_onnx.py` 生成"，但该命令默认只重建
   default 模型；重建强力档需
   `python tools/export_high_removal_onnx.py --config configs/onnx_strong.yaml --output models/high_removal_strong.onnx`。
   仅提示文案问题，三档共享。
3. `tools/metrics_strong.md` 里 16.png 的 default 一行（43 946 / 11.435 / 0.2987）与本环境
   复测（43 950 / 11.436 / 0.2988）有 ≤4 像素级出入——同会话内重复运行逐位可复现，
   属 mediapipe/opencv 版本级微差，不影响任何排序结论；strong 一行数字完全一致。

## 修复清单

无。强力路径（模型、内嵌配置、精确内核 parity、强度排序、面部保护、CLI/Studio 接线）
全部实测通过，未做任何代码/参数改动。
