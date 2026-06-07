"""GPU concurrency gate (P0-1).

Every video job competes for one GPU. Without a gate, N concurrent uploads each
spawn a worker that hits the GPU simultaneously → OOM, and the per-job latency
budget (computed for one job in isolation) becomes meaningless.

A bounded semaphore turns "everyone time-slices the GPU" into "an orderly line".
Set ``GPU_SLOTS`` to the number of jobs that *actually* fit in GPU memory at once
(start at 1, measure with all models loaded before raising). Only the GPU stages
(detect/track/pose/annotate) need the gate — decode/encode/IO stay parallel.

This is the single-box bridge; the real queue + admission policy arrives with the
task queue (P1-2), where ``GPU_SLOTS`` becomes worker ``--concurrency``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, TypeVar

GPU_SLOTS = int(os.getenv("GPU_SLOTS", "1"))
_gpu_sem = threading.BoundedSemaphore(GPU_SLOTS)

T = TypeVar("T")


class GpuBusyError(RuntimeError):
    """Raised when a job can't get a GPU slot within the admission timeout."""


def run_on_gpu(fn: Callable[..., T], *args: Any, timeout_s: float = 180.0,
               **kwargs: Any) -> T:
    """Run ``fn`` holding a GPU slot; raise :class:`GpuBusyError` on admission timeout."""
    acquired = _gpu_sem.acquire(timeout=timeout_s)
    if not acquired:
        raise GpuBusyError(f"GPU busy: admission timeout after {timeout_s}s")
    try:
        return fn(*args, **kwargs)
    finally:
        _gpu_sem.release()


def available_slots() -> int:
    """Best-effort current free slots (diagnostics only)."""
    return _gpu_sem._value  # type: ignore[attr-defined]
