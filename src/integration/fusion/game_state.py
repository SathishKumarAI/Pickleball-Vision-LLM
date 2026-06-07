"""Game-state fusion.

Turns raw per-frame vision outputs (object detections + ball track) into a
structured, JSON-serialisable game state that the LLM layer can reason over.

Pure-Python on purpose — no torch/cv2/numpy — so it runs anywhere and is unit
testable without the vision stack. Detection dicts follow the schema produced by
``src.vision.detection.detector.ObjectDetector.detect``::

    {"bbox": [x1, y1, x2, y2], "confidence": float,
     "class_id": int, "class_name": str}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# COCO class names the YOLO model emits that we care about.
PLAYER_CLASSES = {"person"}
BALL_CLASSES = {"sports ball", "ball"}


def _centroid(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


@dataclass
class PlayerState:
    bbox: List[float]
    centroid: List[float]
    side: str  # "near" | "far" — relative to the midline
    confidence: float


@dataclass
class BallState:
    centroid: Optional[List[float]] = None
    bbox: Optional[List[float]] = None
    confidence: float = 0.0
    velocity: Optional[List[float]] = None  # [dx, dy] px/frame vs previous state


@dataclass
class GameState:
    frame_index: int
    timestamp: float
    players: List[PlayerState] = field(default_factory=list)
    ball: BallState = field(default_factory=BallState)
    action: str = "unknown"
    num_players: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GameStateBuilder:
    """Build :class:`GameState` objects from frame-level vision outputs.

    Keeps minimal cross-frame memory (previous ball centroid) so it can derive
    ball velocity and a coarse action label.
    """

    def __init__(self, frame_height: Optional[int] = None, min_confidence: float = 0.5):
        """
        Args:
            frame_height: Pixel height of the frame, used to split the court into
                near/far halves. If None, side is left "unknown".
            min_confidence: Drop detections below this confidence.
        """
        self.frame_height = frame_height
        self.min_confidence = min_confidence
        self._prev_ball_centroid: Optional[List[float]] = None

    def reset(self) -> None:
        """Clear cross-frame memory (call between clips)."""
        self._prev_ball_centroid = None

    def build(self, detections: List[Dict[str, Any]], frame_index: int,
              fps: float = 30.0) -> GameState:
        """Fuse one frame's detections into a structured game state."""
        players: List[PlayerState] = []
        ball = BallState()

        for det in detections:
            if det.get("confidence", 0.0) < self.min_confidence:
                continue
            name = det.get("class_name", "")
            bbox = det["bbox"]
            c = _centroid(bbox)
            if name in PLAYER_CLASSES:
                players.append(PlayerState(
                    bbox=bbox, centroid=c, confidence=det["confidence"],
                    side=self._side(c),
                ))
            elif name in BALL_CLASSES and det["confidence"] >= ball.confidence:
                ball = BallState(centroid=c, bbox=bbox, confidence=det["confidence"])

        ball.velocity = self._ball_velocity(ball.centroid)
        self._prev_ball_centroid = ball.centroid

        state = GameState(
            frame_index=frame_index,
            timestamp=round(frame_index / fps, 3) if fps else 0.0,
            players=players,
            ball=ball,
            num_players=len(players),
        )
        state.action = self._infer_action(state)
        return state

    # -- heuristics ---------------------------------------------------------

    def _side(self, centroid: List[float]) -> str:
        if self.frame_height is None:
            return "unknown"
        return "near" if centroid[1] >= self.frame_height / 2.0 else "far"

    def _ball_velocity(self, centroid: Optional[List[float]]) -> Optional[List[float]]:
        if centroid is None or self._prev_ball_centroid is None:
            return None
        return [round(centroid[0] - self._prev_ball_centroid[0], 2),
                round(centroid[1] - self._prev_ball_centroid[1], 2)]

    def _infer_action(self, state: GameState) -> str:
        """Coarse action label from ball presence/motion and player count.

        Deliberately simple and explainable — a placeholder for a learned
        action classifier (see docs/PLAN.md, Phase 3). Returns one of:
        no-ball / serve-or-reset / rally / fast-exchange.
        """
        if state.ball.centroid is None:
            return "no-ball"
        v = state.ball.velocity
        if v is None:
            return "serve-or-reset"
        speed = (v[0] ** 2 + v[1] ** 2) ** 0.5
        if speed < 5:
            return "serve-or-reset"
        if speed < 25:
            return "rally"
        return "fast-exchange"
