"""Roboflow detector adapter — pretrained pickleball weights via the OSS
``inference`` SDK, interface-compatible with ``ObjectDetector.detect``.

Lets us swap the generic COCO YOLO for a Roboflow Universe pickleball model
(PB/paddle/ball/player) without touching callers. Lazy-imports ``inference`` so
this module loads without the ``[vision]`` extras; the model runs on the GPU box.

See docs/MODELS_AND_REUSE.md for weight choices (GameChangerv1, ak-zcxgt, Liberin).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class RoboflowDetector:
    """Detect via a Roboflow-hosted/edge model; emit our detection schema."""

    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None,
                 confidence: float = 0.4):
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID", "")
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.confidence = confidence
        self._model = None

    def _ensure(self):
        if self._model is None:
            from inference import get_model  # lazy (OSS Roboflow SDK)
            self._model = get_model(model_id=self.model_id, api_key=self.api_key)
        return self._model

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Run detection on a BGR/RGB numpy frame -> list of detection dicts."""
        model = self._ensure()
        result = model.infer(frame, confidence=self.confidence)[0]
        out: List[Dict[str, Any]] = []
        for p in getattr(result, "predictions", []):
            # Roboflow gives center x,y,w,h -> convert to xyxy.
            x1 = p.x - p.width / 2
            y1 = p.y - p.height / 2
            out.append({
                "bbox": [int(x1), int(y1), int(x1 + p.width), int(y1 + p.height)],
                "confidence": float(p.confidence),
                "class_id": int(getattr(p, "class_id", 0)),
                "class_name": str(p.class_name),
            })
        return out


def get_detector(backend: str = "ultralytics", config: Any = None, **kwargs):
    """Factory: ``ultralytics`` (default, local YOLO) or ``roboflow`` (hosted weights)."""
    if backend == "ultralytics":
        from src.vision.detection.detector import ObjectDetector
        return ObjectDetector(config)
    if backend == "roboflow":
        return RoboflowDetector(**kwargs)
    raise ValueError(f"Unknown detector backend: {backend!r}")
