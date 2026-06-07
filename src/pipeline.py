"""End-to-end analysis pipeline.

Clip ──▶ frames ──▶ detector (YOLO) ──▶ ball tracker ──▶ game-state fusion
      ──▶ coaching feedback (LLM/rule).

The heavy vision deps (cv2, torch, ultralytics) are imported lazily inside
:meth:`Pipeline.process_video`, so this module — and the pure-data path
:meth:`Pipeline.analyze_detections` — import and run without the ``[vision]``
extras. That keeps the fusion + feedback glue testable offline; only real model
inference needs the full stack (run on a GPU machine).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.integration.fusion.game_state import GameStateBuilder
from src.llm.generate_feedback import CoachingFeedbackGenerator


class Pipeline:
    """Orchestrates vision → fusion → feedback."""

    def __init__(self, config: Optional[Any] = None, feedback_backend: str = "rule",
                 tracker_backend: str = "simple", frame_height: Optional[int] = None):
        self.config = config
        self.feedback_backend = feedback_backend
        self.tracker_backend = tracker_backend
        self._frame_height = frame_height
        self._detector = None  # lazy
        self._tracker = None    # lazy
        self.feedback = CoachingFeedbackGenerator(backend=feedback_backend)

    # -- pure-data path (no vision deps) -----------------------------------

    def analyze_detections(self, detections_per_frame: List[List[Dict[str, Any]]],
                           fps: float = 30.0,
                           frame_height: Optional[int] = None) -> Dict[str, Any]:
        """Run fusion + feedback over pre-computed per-frame detections.

        Lets the whole back half of the pipeline be exercised without running
        a model — pass detections from a fixture, a cache, or a real detector.

        Returns:
            ``{"states": [...], "feedback": [...], "summary": str}``
        """
        builder = GameStateBuilder(
            frame_height=frame_height if frame_height is not None else self._frame_height,
            min_confidence=getattr(self.config, "MIN_CONFIDENCE", 0.5),
        )
        states: List[Dict[str, Any]] = []
        feedback: List[str] = []
        for i, dets in enumerate(detections_per_frame):
            state = builder.build(dets, frame_index=i, fps=fps).to_dict()
            states.append(state)
            feedback.append(self.feedback.from_game_state(state))
        summary = self.feedback.summarize(states) if states else ""
        return {"states": states, "feedback": feedback, "summary": summary}

    # -- full vision path (needs [vision] extras) --------------------------

    def _ensure_vision(self) -> None:
        if self._detector is not None:
            return
        # Lazy: only import the heavy stack when actually running a video.
        from src.vision.detection.detector import ObjectDetector
        from src.vision.tracking.tracker import get_tracker
        self._detector = ObjectDetector(self.config)
        self._tracker = get_tracker(self.tracker_backend)

    def process_video(self, video_path: str, frame_skip: int = 5,
                      max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Decode a video, detect per frame, then fuse + generate feedback.

        Requires the ``[vision]`` extras (cv2, torch, ultralytics). Run on a
        machine with the model stack installed.
        """
        import cv2  # lazy

        self._ensure_vision()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        self._frame_height = height

        detections_per_frame: List[List[Dict[str, Any]]] = []
        idx = 0
        kept = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_skip == 0:
                detections_per_frame.append(self._detector.detect(frame))
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            idx += 1
        cap.release()

        result = self.analyze_detections(detections_per_frame, fps=fps, frame_height=height)
        result["frames_processed"] = kept
        result["video"] = video_path
        return result


def analyze_video(video_path: str, config: Optional[Any] = None,
                  feedback_backend: str = "rule") -> Dict[str, Any]:
    """Convenience entry point: full analysis of a video file."""
    return Pipeline(config=config, feedback_backend=feedback_backend).process_video(video_path)
