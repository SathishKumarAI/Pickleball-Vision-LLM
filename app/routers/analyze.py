"""Analyze — the no-GPU fusion + coaching path.

Reuses ``src.pipeline.Pipeline.analyze_detections`` (pure-data; no torch/cv2), so
clients that already have detections get game-state + coaching without the GPU.
Requires auth (it's a real product endpoint).
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.deps import get_current_user
from app.models import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """Fuse pre-computed detections into game state + coaching feedback."""
    from src.pipeline import Pipeline  # lazy: keeps app import light

    pipe = Pipeline(feedback_backend=req.backend, frame_height=req.frame_height)
    return pipe.analyze_detections(
        req.detections, fps=req.fps, frame_height=req.frame_height
    )
