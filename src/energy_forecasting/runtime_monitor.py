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


def _cuda_runtime():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


@contextmanager
def measure_runtime(label: str, records: list[dict], **metadata):
    """Measure a stage with CUDA-safe timing and peak-memory counters."""
    torch = _cuda_runtime()
    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rss0 = _rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch is not None:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rss1 = _rss_mb()
        allocated = reserved = float("nan")
        if torch is not None:
            allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        records.append({
            "stage": label,
            "wall_s": elapsed,
            "rss_mb": rss1,
            "rss_delta_mb": rss1 - rss0,
            "cuda_available": torch is not None,
            "gpu_peak_allocated_mb": allocated,
            "gpu_peak_reserved_mb": reserved,
            **metadata,
        })
