"""Modal GPU worker — deploy with `modal deploy worker/modal_app.py`.

Wraps the vision `Pipeline.process_video` on a GPU, driven by the pure
`worker.runner.run_job` seam. The API spawns `run_analysis` (see
`app.services.dispatch.ModalDispatcher`); progress is streamed back by PATCHing
the Supabase `jobs` row, which Supabase Realtime pushes to the browser.

NOTE: this module imports `modal` at top level, so it only imports where Modal is
installed (the deploy/GPU side). Its control logic is unit-tested via
`worker.runner` + `tests/test_worker_runner.py` without Modal/GPU.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import modal

from worker.runner import JobCancelledError, run_job

# ---------------------------------------------------------------------------
# Image: core + vision + bedrock extras, ffmpeg, and the src/ + app/ + worker/ code.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["vision", "bedrock"])
    .add_local_python_source("src", "app", "worker")
)

app = modal.App("pickleball-gpu", image=image)

# Cache model weights across cold starts (YOLO / Roboflow / TrackNet).
weights = modal.Volume.from_name("pvllm-weights", create_if_missing=True)

# Secrets: Supabase URL + service key, AWS creds for Bedrock.
secrets = [modal.Secret.from_name("pvllm-secrets")]


def _build_analyze_fn(weights_dir: str):
    """Return an analyze_fn for run_job that runs the real vision Pipeline."""
    def analyze(local_in, local_out, *, progress_cb, cancel_cb,
                tracker, feedback_backend, max_seconds):
        # Build the vision Config HERE only — it validates weight files on disk
        # (src/core/config/config.py:_validate_settings raises if absent), which
        # must never run in the CPU API container.
        from src.core.config.config import Config
        from src.pipeline import Pipeline
        from src.services.jobs import JobCancelled

        cfg = Config()
        # Point the detector at the cached weight on the Volume if provided.
        model_path = os.getenv("DETECTOR_MODEL")
        if model_path:
            cfg.MODEL_PATH = model_path  # consumed by detector._load_model fallback

        pipe = Pipeline(config=cfg, feedback_backend=feedback_backend, tracker_backend=tracker)
        try:
            return pipe.process_video(
                local_in, annotate_path=local_out,
                progress_cb=progress_cb, cancel_cb=cancel_cb, max_seconds=max_seconds,
            )
        except JobCancelled as e:  # translate to the runner's cancel signal
            raise JobCancelledError() from e
    return analyze


@app.function(
    gpu="A10G",
    volumes={"/weights": weights},
    secrets=secrets,
    timeout=900,
    retries=modal.Retries(max_retries=2),
    max_containers=2,   # GPU admission cap (replaces the old semaphore gpu_gate)
)
def run_analysis(job_id: str, user_id: str, input_object_key: str,
                 tracker: str = "supervision", feedback_backend: str = "rule"):
    """Modal entry point — one analysis job on a GPU."""
    from app.config import Settings
    from app.services.repo import SupabaseRepo
    from app.services.storage import SupabaseStorage

    s = Settings()
    repo = SupabaseRepo(s.supabase_url, s.supabase_service_key)
    storage = SupabaseStorage(s.supabase_url, s.supabase_service_key)
    period = datetime.now(timezone.utc).strftime("%Y-%m")

    return run_job(
        job_id, user_id, input_object_key,
        repo=repo, storage=storage,
        analyze_fn=_build_analyze_fn("/weights"),
        uploads_bucket=s.uploads_bucket, outputs_bucket=s.outputs_bucket,
        period=period, tracker=tracker, feedback_backend=feedback_backend,
    )


@app.function(secrets=secrets, schedule=modal.Period(days=1))
def retention_sweep(retention_days: int = 30):
    """Daily cron: delete annotated videos + job rows older than N days (GDPR/cost).

    Storage objects also expire via a Supabase Storage lifecycle rule; this sweep
    is the belt-and-suspenders DB + output cleanup.
    """
    from datetime import datetime, timedelta, timezone

    from app.config import Settings
    from supabase import create_client

    s = Settings()
    sb = create_client(s.supabase_url, s.supabase_service_key)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

    old = sb.table("jobs").select("id,output_video_key,result_object_key") \
        .lt("created_at", cutoff).execute().data
    for job in old:
        for key in (job.get("output_video_key"), job.get("result_object_key")):
            if key:
                try:
                    sb.storage.from_(s.outputs_bucket).remove([key])
                except Exception:  # noqa: BLE001
                    pass
        sb.table("jobs").delete().eq("id", job["id"]).execute()
    return {"deleted_jobs": len(old)}


@app.local_entrypoint()
def main(job_id: str = "test", user_id: str = "test", key: str = "test/in.mp4"):
    """Manual trigger for the Phase-1 seam test: `modal run worker/modal_app.py`."""
    print(run_analysis.remote(job_id, user_id, key))
