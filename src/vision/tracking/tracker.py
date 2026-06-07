"""Ball / object tracking across frames.

Two backends, selected via :func:`get_tracker`:

* ``simple``     — zero-dependency nearest/highest-confidence tracker. Always
  available; good enough for the offline path and tests.
* ``supervision``— Roboflow **supervision** ByteTrack (lazy-imported). Preferred
  for real runs: stable track IDs, motion continuity. Don't reinvent ByteTrack.

Heavy deps (numpy, supervision) are imported lazily, so this module imports and
the ``simple`` backend runs without the ``[vision]`` extras installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BallTracker:
    """Simple, dependency-free tracker.

    Picks the highest-confidence ball detection per frame and keeps a history.
    A pragmatic baseline — for robust small-fast-ball tracking prefer a TrackNet
    backend (see docs/MODELS_AND_REUSE.md).
    """

    def __init__(self) -> None:
        self.track_history: List[Dict[str, Any]] = []
        self.last_position: Optional[Dict[str, Any]] = None

    def update(self, detections: List[Dict[str, Any]], frame: Any = None) -> Optional[Dict[str, Any]]:
        """Return the most likely ball detection for this frame."""
        if not detections:
            return None
        best = max(detections, key=lambda d: d.get("confidence", 0.0))
        self.last_position = best
        self.track_history.append(best)
        return best

    def predict_next_position(self) -> Optional[Dict[str, Any]]:
        """Linear extrapolation from the last two tracked centroids."""
        if len(self.track_history) < 2:
            return None
        (ax1, ay1, ax2, ay2) = self.track_history[-2]["bbox"]
        (bx1, by1, bx2, by2) = self.track_history[-1]["bbox"]
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        return {"x": bcx + (bcx - acx), "y": bcy + (bcy - acy)}


class SupervisionTracker:
    """ByteTrack-backed multi-object tracker via Roboflow ``supervision``.

    Assigns stable ``tracker_id``s across frames. Lazy-imports supervision so the
    module stays importable without the extra; raises a clear error only when
    actually used without it installed.
    """

    def __init__(self, **byte_track_kwargs: Any) -> None:
        self._kwargs = byte_track_kwargs
        self._tracker = None  # lazy

    def _ensure(self) -> None:
        if self._tracker is not None:
            return
        try:
            import supervision as sv
        except ImportError as e:  # pragma: no cover - needs [vision] extra
            raise RuntimeError(
                "supervision backend needs `pip install supervision`"
            ) from e
        self._sv = sv
        self._tracker = sv.ByteTrack(**self._kwargs)

    def update(self, detections: List[Dict[str, Any]], frame: Any = None) -> List[Dict[str, Any]]:
        """Update tracks; returns detections enriched with ``tracker_id``.

        Args:
            detections: detector output (bbox/confidence/class_id/class_name).
            frame: unused (kept for interface parity).
        """
        self._ensure()
        import numpy as np

        if not detections:
            return []
        sv = self._sv
        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        conf = np.array([d.get("confidence", 0.0) for d in detections], dtype=float)
        cls = np.array([d.get("class_id", 0) for d in detections], dtype=int)
        sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = self._tracker.update_with_detections(sv_dets)

        out: List[Dict[str, Any]] = []
        for i in range(len(tracked)):
            out.append({
                "bbox": tracked.xyxy[i].tolist(),
                "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                "class_id": int(tracked.class_id[i]) if tracked.class_id is not None else 0,
                "tracker_id": int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1,
            })
        return out


def get_tracker(backend: str = "simple", **kwargs: Any):
    """Factory: ``simple`` (no deps) or ``supervision`` (ByteTrack)."""
    if backend == "simple":
        return BallTracker()
    if backend == "supervision":
        return SupervisionTracker(**kwargs)
    raise ValueError(f"Unknown tracker backend: {backend!r}")
