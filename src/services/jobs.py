"""Async job orchestration for long-running video analysis.

In-process, thread-safe job store + worker. This is the control plane for the
"upload video → annotated output in minutes" product: the API submits a job and
returns immediately; a background thread runs the (potentially minutes-long)
processor and streams progress into the job; the client polls for completion.

Dependency-free on purpose — the processor is injected as a callable, so the
whole lifecycle is unit-testable without the ML stack. In production the
processor is ``src.pipeline.Pipeline.process_video``. Swap this store for
Redis/Celery when scaling past one box (see the architecture note).
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class JobCancelled(Exception):
    """Raised by a processor that observed a cancel request and stopped."""


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0..1
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
        }


# Processor signature: (job, progress_cb) -> result dict.
# progress_cb(fraction: float, message: str) updates the job as work proceeds.
Processor = Callable[["Job", Callable[[float, str], None]], Dict[str, Any]]


class JobStore:
    """Thread-safe job registry + background runner."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, meta: Optional[Dict[str, Any]] = None) -> Job:
        job = Job(id=uuid.uuid4().hex, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def request_cancel(self, job_id: str) -> bool:
        """Mark a job for cooperative cancellation. Returns False if not cancellable."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in (JobStatus.DONE, JobStatus.ERROR,
                                             JobStatus.CANCELLED):
                return False
            job.status = JobStatus.CANCELLING
            return True

    def is_cancelling(self, job_id: str) -> bool:
        """Worker polls this each batch to honour a cancel request."""
        job = self.get(job_id)
        return bool(job and job.status == JobStatus.CANCELLING)

    def submit(self, job: Job, processor: Processor, *, sync: bool = False) -> Job:
        """Run ``processor`` for ``job``. Async (daemon thread) unless ``sync``.

        ``sync=True`` runs inline — used by tests for deterministic lifecycle.
        """
        def run() -> None:
            # Honour a cancel requested while still queued (don't clobber it with RUNNING).
            if job.status == JobStatus.CANCELLING:
                self._set(job, status=JobStatus.CANCELLED, message="cancelled")
                return
            self._set(job, status=JobStatus.RUNNING, progress=0.0, message="started")

            def progress_cb(frac: float, msg: str = "") -> None:
                self._set(job, progress=max(0.0, min(1.0, frac)), message=msg)

            try:
                result = processor(job, progress_cb)
                # A cancel requested mid-run wins over a late completion.
                if job.status == JobStatus.CANCELLING:
                    self._set(job, status=JobStatus.CANCELLED, message="cancelled")
                else:
                    self._set(job, status=JobStatus.DONE, progress=1.0,
                              message="completed", result=result)
            except JobCancelled:
                self._set(job, status=JobStatus.CANCELLED, message="cancelled")
            except Exception as e:  # noqa: BLE001 - surface any worker failure
                self._set(job, status=JobStatus.ERROR,
                          error=f"{type(e).__name__}: {e}",
                          message="failed")
                job.meta["traceback"] = traceback.format_exc()

        if sync:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()
        return job

    # -- internal ----------------------------------------------------------

    def _set(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)


# Module-level singleton used by the API blueprint.
store = JobStore()
