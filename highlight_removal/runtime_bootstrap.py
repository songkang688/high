from __future__ import annotations

import gc
import os

FORCE_CPU_ONLY = True
DEFAULT_RUNTIME_MODE = "CPU狂暴模式"
CPU_RUNTIME_MODES = frozenset(
    {
        "CPU正常模式",
        "CPU狂暴模式",
        "CPU低级模式（单线程）",
    }
)


def logical_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 8))


def block_gpu_devices() -> None:
    """Hide GPU devices from CUDA/DirectML stacks; keep CPU-only execution."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_max_thread_env(threads: int | None = None) -> int:
    n = max(1, int(threads or logical_cpu_count()))
    v = str(n)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        os.environ[key] = v
    return n


def boost_memory_policy() -> None:
    """Prefer throughput over aggressive GC; allow larger in-process caches."""
    gc.set_threshold(2000, 20, 20)


def normalize_runtime_mode(mode: str | None) -> str:
    raw = str(mode or DEFAULT_RUNTIME_MODE)
    if raw == "GPU模式" or raw not in CPU_RUNTIME_MODES:
        return DEFAULT_RUNTIME_MODE
    return raw


def bootstrap_before_numpy() -> int:
    """Call before importing cv2/numpy in every app entry point."""
    block_gpu_devices()
    boost_memory_policy()
    return set_max_thread_env(logical_cpu_count())
