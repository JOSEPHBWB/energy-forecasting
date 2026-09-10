from __future__ import annotations

import os
import time
from contextlib import contextmanager

try:
    import psutil
except ImportError:
    psutil = None


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _torch_cuda():
    """Return (torch, cuda_available) without making torch a core dependency."""
    try:
        import torch
    except ImportError:
        return None, False
    return torch, bool(torch.cuda.is_available())


@contextmanager
def profile_stage(stage: str, sink: list[dict]):
    """Profile wall time, process RSS, and CUDA peak memory when available.

    CUDA operations are synchronized before and after the measured stage so
    asynchronous kernels do not make the wall-clock measurement artificially
    small. Peak-memory fields are NaN on CPU-only machines.
    """
    torch, cuda_available = _torch_cuda()
    if cuda_available:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda_available:
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        rss_after = _rss_mb()

        peak_allocated_mb = float("nan")
        peak_reserved_mb = float("nan")
        if cuda_available:
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        sink.append(
            {
                "stage": stage,
                "wall_s": wall_s,
                "rss_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "cuda_available": cuda_available,
                "gpu_peak_allocated_mb": peak_allocated_mb,
                "gpu_peak_reserved_mb": peak_reserved_mb,
            }
        )
