"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Run the no-GPU fusion+feedback path over pre-computed detections."""
    detections: List[List[Dict[str, Any]]] = Field(
        ..., description="Per-frame detection lists (bbox/confidence/class_id/class_name)."
    )
    fps: float = 30.0
    frame_height: Optional[int] = None
    backend: str = "rule"


class AnalyzeResponse(BaseModel):
    states: List[Dict[str, Any]]
    feedback: List[str]
    summary: str


class UserOut(BaseModel):
    id: Optional[str]
    email: Optional[str]
    role: str = "authenticated"
    is_admin: bool = False


# --- jobs ---
class UploadUrlRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


class UploadUrlResponse(BaseModel):
    object_key: str
    upload_url: str
    bucket: str


class JobCreate(BaseModel):
    object_key: str = Field(..., description="Storage key of the uploaded video.")
    content_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_s: Optional[float] = None
    tracker: str = "supervision"
    backend: str = "rule"


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    deduplicated: bool = False


class QuotaOut(BaseModel):
    plan: str
    used: int
    limit: int
    remaining: int


class JobOut(BaseModel):
    id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = ""
    error: Optional[str] = None
    tracker: Optional[str] = None
    backend: Optional[str] = None
    created_at: Optional[str] = None


class JobListOut(BaseModel):
    jobs: List[JobOut]
    count: int
