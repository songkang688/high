# Windows 预编译自定义算子 DLL（x64）

由 GitHub Actions workflow [`windows-dll`](../../../.github/workflows/windows-dll.yml)
在 `windows-latest`（windows-2025-vs2026 镜像，MSVC 14.51）上构建，构建配置：
vcpkg `x64-windows-static-md`（OpenCV 4.12 / yaml-cpp 静态链接进 DLL）+
onnxruntime-win-x64-1.22.0 头文件 + Python 3.12（`high_removal_pyops.dll` 走
CPython 稳定 ABI，运行时兼容任意 Python ≥ 3.10）。

本副本来自构建
[songkang688/high actions run 32876643068](https://github.com/songkang688/high/actions/runs/32876643068)，
该 run 的冒烟测试在 Windows 真机上验证过：精确内核输出与 `process_image` **逐位相同**，
C++ 内核单图 MAE 0.00457/255。

| 文件 | SHA256 | 运行期依赖 |
|------|--------|-----------|
| `high_removal_ops.dll`（C++ 快速内核，4.4 MB） | `e8f79babd0cb7cb95a2a29a83f983b89c6dca7380fe5b1e29999ed354582e69b` | 仅 MSVC 运行库（VC++ 2015-2022 Redistributable x64）；OpenCV/yaml-cpp 已静态链接 |
| `high_removal_pyops.dll`（精确内核，48 KB） | `74590f9445c2195e58471174aefd539c573bebea4d326667f6e6cad028f9897b` | 宿主 CPython ≥ 3.10 的 `python3.dll` + MSVC 运行库；宿主须为 Python 进程且本仓库可导入 |

`onnxruntime.dll` 由宿主提供（pip 包 onnxruntime 或 C++ 应用自带），两个 DLL 均不携带。
使用方式见 `cpp/README.md`；`tools/run_high_onnx.py` 在 Windows 上会自动找到本目录。
