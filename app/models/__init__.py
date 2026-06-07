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
