"""Pure orchestration of the GPU analysis seam.

This is the logic that runs inside the Modal function, factored out so it can be
unit-tested without Modal/GPU/cv2: the model step is injected as ``analyze_fn``.
It owns idempotency, download → analyze → upload, throttled progress, cooperative
cancel, error capture, and usage metering.

In production (``worker/modal_app.py``) ``analyze_fn`` is
``Pipeline(...).process_video`` and repo/storage are the Supabase impls.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from app.services.repo import Repo
from app.services.storage import Storage

# analyze_fn(local_in, local_out, progress_cb, cancel_cb) -> result dict
AnalyzeFn = Callable[..., Dict[str, Any]]


class _Throttle:
    """Emit at most one progress update per ``min_delta`` fraction change."""

    def __init__(self, min_delta: float = 0.02):
        self.min_delta = min_delta
        self._last = -1.0

    def should(self, frac: float) -> bool:
        if frac >= 1.0 or frac - self._last >= self.min_delta:
            self._last = frac
            return True
        return False


class JobCancelledError(Exception):
    pass


def run_job(
    job_id: str,
    user_id: str,
    input_object_key: str,
    *,
    repo: Repo,
    storage: Storage,
    analyze_fn: AnalyzeFn,
    uploads_bucket: str,
    outputs_bucket: str,
    period: str,
    workdir: str = "/tmp",
    tracker: str = "supervision",
    feedback_backend: str = "rule",
    max_seconds: float = 450.0,
) -> Dict[str, Any]:
    """Run one analysis job end-to-end. Returns the final job row.

    Idempotent: if the job isn't claimable (already running/done), returns early.
    """
    # 1) Idempotency — atomically claim queued -> running.
    if not repo.claim_running(job_id):
        existing = repo.get_job(job_id)
        return existing or {"id": job_id, "status": "skipped"}

    local_in = os.path.join(workdir, f"{job_id}_in.mp4")
    local_out = os.path.join(workdir, f"{job_id}_out.mp4")
    throttle = _Throttle()

    def progress_cb(frac: float, msg: str = "") -> None:
        if throttle.should(frac):
            repo.update_job(job_id, progress=round(float(frac), 3), message=msg)

    def cancel_cb() -> bool:
        j = repo.get_job(job_id)
        return bool(j and j.get("status") == "cancelling")

    try:
        storage.download_file(uploads_bucket, input_object_key, local_in)
        progress_cb(0.05, "downloaded")

        result = analyze_fn(
            local_in, local_out,
            progress_cb=lambda f, m="processing": progress_cb(0.05 + 0.9 * f, m),
            cancel_cb=cancel_cb,
            tracker=tracker, feedback_backend=feedback_backend, max_seconds=max_seconds,
        )

        # Cancelled mid-run wins over a late completion.
        if cancel_cb():
            return repo.update_job(job_id, status="cancelled", message="cancelled")

        out_key = f"{user_id}/{job_id}/annotated.mp4"
        res_key = f"{user_id}/{job_id}/result.json"
        if os.path.exists(local_out):
            storage.upload_file(outputs_bucket, out_key, local_out)
        else:
            out_key = None

        seconds = int(result.get("duration_s") or 0)
        repo.incr_usage(user_id, period, videos=1, seconds=seconds)

        return repo.update_job(
            job_id, status="done", progress=1.0, message="completed",
            output_video_key=out_key, result_object_key=res_key,
            summary=result.get("summary"),
            frames_processed=result.get("frames_processed"),
            duration_s=result.get("duration_s"),
        )
    except JobCancelledError:
        return repo.update_job(job_id, status="cancelled", message="cancelled")
    except Exception as e:  # noqa: BLE001 - surface any worker failure
        return repo.update_job(job_id, status="error",
                               error=f"{type(e).__name__}: {e}", message="failed")
    finally:
        for p in (local_in, local_out):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
