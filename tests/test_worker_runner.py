"""Phase 1 offline tests: the worker seam (worker.runner.run_job) with a fake
GPU analyze function — exercises idempotency, progress, cancel, error, usage,
without Modal/GPU/cv2."""

import os

import pytest

from app.services.repo import InMemoryRepo
from app.services.storage import FakeStorage
from worker.runner import JobCancelledError, run_job

PERIOD = "2026-06"


def _seed_job(repo, status="queued", **extra):
    return repo.create_job("u1", input_object_key="u1/x/in.mp4", status=status, **extra)


def _ok_analyze(local_in, local_out, *, progress_cb, cancel_cb, tracker, feedback_backend, max_seconds):
    progress_cb(0.5, "half")
    # simulate the encoder writing an output file
    with open(local_out, "wb") as f:
        f.write(b"fake-mp4")
    return {"summary": "nice rally", "frames_processed": 120, "duration_s": 4.0,
            "states": [], "feedback": []}


def _common(repo, storage, analyze_fn, job):
    return run_job(job["id"], "u1", "u1/x/in.mp4", repo=repo, storage=storage,
                   analyze_fn=analyze_fn, uploads_bucket="uploads", outputs_bucket="outputs",
                   period=PERIOD, workdir="/tmp")


def test_success_path():
    repo, storage = InMemoryRepo(), FakeStorage()
    job = _seed_job(repo)
    out = _common(repo, storage, _ok_analyze, job)
    assert out["status"] == "done"
    assert out["progress"] == 1.0 and out["summary"] == "nice rally"
    assert out["output_video_key"] == f"u1/{job['id']}/annotated.mp4"
    # usage metered
    assert repo.get_usage("u1", PERIOD)["videos_processed"] == 1
    assert repo.get_usage("u1", PERIOD)["seconds_processed"] == 4
    # output uploaded; temp files cleaned
    assert f"outputs/u1/{job['id']}/annotated.mp4" in storage.uploaded
    assert not os.path.exists(f"/tmp/{job['id']}_in.mp4")


def test_idempotency_already_claimed():
    repo, storage = InMemoryRepo(), FakeStorage()
    job = _seed_job(repo, status="running")  # not claimable
    calls = []
    def analyze(*a, **k):
        calls.append(1); return {}
    out = _common(repo, storage, analyze, job)
    assert not calls            # analyze never ran
    assert out["status"] == "running"


def test_cancel_midrun():
    repo, storage = InMemoryRepo(), FakeStorage()
    job = _seed_job(repo)
    def analyze(local_in, local_out, *, progress_cb, cancel_cb, **k):
        repo.update_job(job["id"], status="cancelling")  # user cancels during run
        return {"duration_s": 1}
    out = _common(repo, storage, analyze, job)
    assert out["status"] == "cancelled"
    assert repo.get_usage("u1", PERIOD)["videos_processed"] == 0  # not metered


def test_cancel_raised():
    repo, storage = InMemoryRepo(), FakeStorage()
    job = _seed_job(repo)
    def analyze(*a, **k):
        raise JobCancelledError()
    out = _common(repo, storage, analyze, job)
    assert out["status"] == "cancelled"


def test_error_capture():
    repo, storage = InMemoryRepo(), FakeStorage()
    job = _seed_job(repo)
    def analyze(*a, **k):
        raise RuntimeError("cuda oom")
    out = _common(repo, storage, analyze, job)
    assert out["status"] == "error"
    assert "RuntimeError: cuda oom" in out["error"]
    assert repo.get_usage("u1", PERIOD)["videos_processed"] == 0
