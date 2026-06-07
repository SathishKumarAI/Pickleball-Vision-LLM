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

import time
from typing import Any, Callable, Dict, List, Optional

from src.integration.fusion.game_state import GameStateBuilder
from src.llm.generate_feedback import CoachingFeedbackGenerator
from src.services.gpu_gate import run_on_gpu
from src.services.jobs import JobCancelled


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
                           frame_height: Optional[int] = None,
                           homography: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """Run fusion + feedback (+ court analytics) over per-frame detections.

        Lets the whole back half of the pipeline be exercised without running
        a model — pass detections from a fixture, a cache, or a real detector.

        Returns:
            ``{"states": [...], "feedback": [...], "summary": str, "metrics": {...}}``
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

        from src.integration.analytics import compute_match_metrics  # lazy (numpy)
        metrics = compute_match_metrics(
            states, fps=fps, homography=homography, frame_h=frame_height
        )
        return {"states": states, "feedback": feedback, "summary": summary, "metrics": metrics}

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
                      max_frames: Optional[int] = None,
                      annotate_path: Optional[str] = None,
                      progress_cb: Optional[Any] = None,
                      cancel_cb: Optional[Callable[[], bool]] = None,
                      max_seconds: Optional[float] = None,
                      gpu_timeout_s: float = 180.0) -> Dict[str, Any]:
        """Decode a video, detect per frame, fuse + generate feedback, and
        optionally write an annotated output video.

        Args:
            frame_skip: process 1 of every N frames (latency lever).
            max_frames: cap processed frames.
            annotate_path: if set, write an annotated ``.mp4`` here (boxes/IDs via
                supervision when available, else plain OpenCV boxes).
            progress_cb: optional ``callable(fraction_0_1, message)`` for job UIs.

        Requires the ``[vision]`` extras (cv2, torch, ultralytics). Run on a
        machine with the model stack installed.
        """
        import cv2  # lazy

        self._ensure_vision()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        self._frame_height = height

        writer = None
        annotator = self._make_annotator() if annotate_path else None
        if annotate_path and width and height:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotate_path, fourcc, fps, (width, height))

        detections_per_frame: List[List[Dict[str, Any]]] = []
        idx = 0
        kept = 0
        last_annot = None  # hold last annotated frame for skipped frames
        started = time.monotonic()
        while cap.isOpened():
            # P0-4: cooperative cancel + whole-job wall-clock ceiling.
            if cancel_cb and cancel_cb():
                cap.release()
                if writer is not None:
                    writer.release()
                raise JobCancelled("cancelled by request")
            if max_seconds and (time.monotonic() - started) > max_seconds:
                cap.release()
                if writer is not None:
                    writer.release()
                raise TimeoutError(f"job exceeded max_seconds={max_seconds}")
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_skip == 0:
                # P0-1: serialize GPU access so concurrent jobs can't OOM the device.
                dets = run_on_gpu(self._detector.detect, frame, timeout_s=gpu_timeout_s)
                detections_per_frame.append(dets)
                kept += 1
                if writer is not None:
                    last_annot = annotator(frame, dets)
                if progress_cb and total:
                    progress_cb(min(0.99, idx / total), f"frame {idx}/{total}")
                if max_frames and kept >= max_frames:
                    if writer is not None:
                        writer.write(last_annot)
                    break
            if writer is not None:
                writer.write(last_annot if last_annot is not None else frame)
            idx += 1
        cap.release()
        if writer is not None:
            writer.release()

        result = self.analyze_detections(detections_per_frame, fps=fps, frame_height=height)
        result["frames_processed"] = kept
        result["video"] = video_path
        if annotate_path:
            result["annotated_video"] = annotate_path
        if progress_cb:
            progress_cb(1.0, "done")
        return result

    def _make_annotator(self):
        """Return ``annotate(frame, detections) -> frame``.

        Prefers supervision annotators (boxes + track IDs); falls back to plain
        OpenCV rectangles when supervision isn't installed.
        """
        import cv2
        try:
            import numpy as np
            import supervision as sv
            box = sv.BoxAnnotator()
            label = sv.LabelAnnotator()

            def annotate(frame, dets):
                if not dets:
                    return frame.copy()
                xyxy = np.array([d["bbox"] for d in dets], dtype=float)
                conf = np.array([d.get("confidence", 0.0) for d in dets], dtype=float)
                cls = np.array([d.get("class_id", 0) for d in dets], dtype=int)
                sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
                labels = [d.get("class_name", str(d.get("class_id", ""))) for d in dets]
                out = box.annotate(frame.copy(), sv_dets)
                return label.annotate(out, sv_dets, labels)
            return annotate
        except ImportError:
            def annotate(frame, dets):
                out = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out, d.get("class_name", ""), (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                return out
            return annotate


def analyze_video(video_path: str, config: Optional[Any] = None,
                  feedback_backend: str = "rule") -> Dict[str, Any]:
    """Convenience entry point: full analysis of a video file."""
    return Pipeline(config=config, feedback_backend=feedback_backend).process_video(video_path)
