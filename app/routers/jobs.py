"""Jobs — async video analysis control plane.

Flow (direct-to-storage upload; API never streams the video):
1. ``POST /jobs/upload-url`` → signed Supabase Storage URL; browser uploads to it.
2. ``POST /jobs`` → quota check → dedup by content hash → create job row → spawn
   the Modal GPU function → returns ``job_id``. The browser then subscribes to the
   job row via Supabase realtime for live progress.
3. ``GET /jobs`` / ``/jobs/{id}`` / ``/result`` / ``/video`` / ``POST /jobs/{id}/cancel``.

Ports the ownership logic from the legacy Flask ``src/api/blueprints/jobs.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException, status

log = logging.getLogger("pvllm.jobs")

from app.config import Settings, get_settings
from app.deps import get_current_user, get_dispatcher, get_repo, get_storage
from app.models import (
    JobCreate, JobCreateResponse, JobListOut, JobOut, QuotaOut,
    UploadUrlRequest, UploadUrlResponse,
)
from app.services.dispatch import Dispatcher
from app.services.quota import check_quota
from app.services.repo import Repo, new_id
from app.services.storage import Storage

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _own(repo: Repo, job_id: str, user: Dict[str, Any]) -> Tuple[Dict[str, Any], None]:
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    if job["user_id"] != user["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")
    return job, None


@router.get("/quota", response_model=QuotaOut)
def get_quota(user=Depends(get_current_user), repo: Repo = Depends(get_repo),
              settings: Settings = Depends(get_settings)):
    sub = repo.get_subscription(user["id"])
    used = repo.get_usage(user["id"], _period())["videos_processed"]
    q = check_quota(sub["plan"], used, settings)
    return QuotaOut(plan=q.plan, used=q.used, limit=q.limit, remaining=q.remaining)


@router.post("/upload-url", response_model=UploadUrlResponse)
def upload_url(req: UploadUrlRequest, user=Depends(get_current_user),
               storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)):
    key = f"{user['id']}/{new_id()}/{req.filename}"
    signed = storage.signed_upload_url(settings.uploads_bucket, key)
    return UploadUrlResponse(object_key=key, upload_url=signed["url"], bucket=settings.uploads_bucket)


@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(req: JobCreate, user=Depends(get_current_user), repo: Repo = Depends(get_repo),
               dispatcher: Dispatcher = Depends(get_dispatcher), settings: Settings = Depends(get_settings)):
    # 1) Quota
    sub = repo.get_subscription(user["id"])
    used = repo.get_usage(user["id"], _period())["videos_processed"]
    q = check_quota(sub["plan"], used, settings)
    if not q.ok:
        log.warning("quota exceeded", extra={"user": user["id"], "plan": q.plan, "used": q.used})
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            detail={"error": "quota exceeded", "plan": q.plan, "used": q.used,
                    "limit": q.limit, "upgrade": True},
        )
    # 2) Idempotency / dedup by content hash
    if req.content_sha256:
        existing = repo.find_job_by_sha(user["id"], req.content_sha256)
        if existing:
            return JobCreateResponse(job_id=existing["id"], status=existing["status"], deduplicated=True)
    # 3) Create job row
    job = repo.create_job(
        user["id"], input_object_key=req.object_key, content_sha256=req.content_sha256,
        duration_s=req.duration_s, tracker=req.tracker, feedback_backend=req.backend,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    # 4) Spawn GPU work
    call_id = dispatcher.spawn({
        "job_id": job["id"], "user_id": user["id"], "input_object_key": req.object_key,
        "tracker": req.tracker, "feedback_backend": req.backend,
    })
    repo.update_job(job["id"], modal_call_id=call_id)
    log.info("job created", extra={"user": user["id"], "job": job["id"], "tracker": req.tracker})
    return JobCreateResponse(job_id=job["id"], status="queued")


@router.get("", response_model=JobListOut)
def list_jobs(user=Depends(get_current_user), repo: Repo = Depends(get_repo)):
    jobs = repo.list_jobs(user["id"])
    return JobListOut(jobs=[JobOut(**_to_out(j)) for j in jobs], count=len(jobs))


@router.get("/{job_id}", response_model=JobOut)
def job_status(job_id: str, user=Depends(get_current_user), repo: Repo = Depends(get_repo)):
    job, _ = _own(repo, job_id, user)
    return JobOut(**_to_out(job))


@router.get("/{job_id}/result")
def job_result(job_id: str, user=Depends(get_current_user), repo: Repo = Depends(get_repo)):
    job, _ = _own(repo, job_id, user)
    if job["status"] != "done":
        raise HTTPException(status.HTTP_409_CONFLICT, detail={"error": "not ready", "status": job["status"]})
    return {"summary": job.get("summary"), "result_object_key": job.get("result_object_key")}


@router.get("/{job_id}/video")
def job_video(job_id: str, user=Depends(get_current_user), repo: Repo = Depends(get_repo),
              storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)):
    job, _ = _own(repo, job_id, user)
    key = job.get("output_video_key")
    if job["status"] != "done" or not key:
        raise HTTPException(status.HTTP_409_CONFLICT, detail={"error": "video not ready", "status": job["status"]})
    return {"url": storage.signed_download_url(settings.outputs_bucket, key)}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, user=Depends(get_current_user), repo: Repo = Depends(get_repo),
               dispatcher: Dispatcher = Depends(get_dispatcher)):
    job, _ = _own(repo, job_id, user)
    if job["status"] in ("done", "error", "cancelled"):
        raise HTTPException(status.HTTP_409_CONFLICT, detail={"error": "not cancellable", "status": job["status"]})
    repo.update_job(job_id, status="cancelling")
    if job.get("modal_call_id"):
        dispatcher.cancel(job["modal_call_id"])
    return {"job_id": job_id, "status": "cancelling"}


def _to_out(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": job["id"], "status": job["status"], "progress": job.get("progress", 0.0),
        "message": job.get("message", ""), "error": job.get("error"),
        "tracker": job.get("tracker"), "backend": job.get("feedback_backend"),
        "created_at": job.get("created_at"),
    }
