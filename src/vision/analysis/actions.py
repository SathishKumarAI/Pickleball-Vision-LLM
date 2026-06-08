"""Shot / action classification (pickleball-specific, OSS-backed).

There is no off-the-shelf pickleball shot classifier, so we build the taxonomy
layer from base — but on top of OSS primitives (scipy features, a **scikit-learn**
model head) rather than hand-rolling ML. The classifier extracts kinematic
features from a ball-trajectory segment and labels the shot using the PB-Vision
aligned taxonomy: ``serve · return · drive · drop · dink · lob · volley``.

Two heads:
* **rule** (default) — transparent, explainable thresholds; needs no training data,
  good baseline + always available.
* **model** — a trained ``sklearn`` classifier (e.g. RandomForest) on the same
  feature vector. Pass one in to upgrade without changing callers.

Features come from :class:`src.vision.analysis.trajectory.TrajectoryStats`
(image coords, y-down). All pure-NumPy; runs and tests offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.vision.analysis.trajectory import TrajectoryStats

SHOT_CLASSES = ["serve", "return", "drive", "drop", "dink", "lob", "volley", "unknown"]


@dataclass
class ShotFeatures:
    mean_speed: float
    max_speed: float
    horiz_travel: float
    vert_span: float       # apex height (max-min y)
    arc_ratio: float       # vert_span / horiz_travel
    descend: float         # end_y - start_y (image y down: + = landing/downward)
    n_bounces: int

    def vector(self) -> List[float]:
        return [self.mean_speed, self.max_speed, self.horiz_travel,
                self.vert_span, self.arc_ratio, self.descend, float(self.n_bounces)]

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def extract_features(stats: TrajectoryStats) -> ShotFeatures:
    """Kinematic features for one shot (a trajectory segment)."""
    path = stats.path
    if len(path) < 2:
        return ShotFeatures(0, 0, 0, 0, 0, 0, len(stats.bounces))
    horiz = float(abs(path[-1, 0] - path[0, 0]))
    return ShotFeatures(
        mean_speed=float(stats.mean_speed),
        max_speed=float(stats.max_speed),
        horiz_travel=horiz,
        vert_span=float(stats.apex_height_px),
        arc_ratio=float(stats.apex_height_px / (horiz + 1e-6)),
        descend=float(path[-1, 1] - path[0, 1]),
        n_bounces=len(stats.bounces),
    )


@dataclass
class ShotThresholds:
    slow: float = 4.0
    moderate: float = 8.0
    fast: float = 14.0
    short_travel: float = 60.0
    high_arc: float = 0.60
    mid_arc: float = 0.25
    flat_arc: float = 0.18


class ShotClassifier:
    """Classify a shot from trajectory features.

    Args:
        model: optional fitted sklearn classifier (``predict``/``predict_proba`` on
            the 7-d feature vector). If given, it takes precedence over the rules.
        thresholds: rule-head thresholds (px/frame + ratios).
    """

    def __init__(self, model: Optional[Any] = None, thresholds: Optional[ShotThresholds] = None):
        self.model = model
        self.t = thresholds or ShotThresholds()

    def classify(self, stats: TrajectoryStats) -> Dict[str, Any]:
        f = extract_features(stats)
        if self.model is not None:
            label, conf = self._model_head(f)
        else:
            label, conf = self._rule_head(f)
        return {"shot": label, "confidence": round(conf, 2), "features": f.to_dict()}

    # -- heads --------------------------------------------------------------

    def _model_head(self, f: ShotFeatures):
        x = np.array([f.vector()])
        label = str(self.model.predict(x)[0])
        conf = 0.75
        if hasattr(self.model, "predict_proba"):
            conf = float(np.max(self.model.predict_proba(x)))
        return label, conf

    def _rule_head(self, f: ShotFeatures):
        t = self.t
        if f.mean_speed < t.slow and f.horiz_travel < t.short_travel:
            return "dink", 0.7
        if f.arc_ratio >= t.high_arc:
            return "lob", 0.75
        if f.max_speed >= t.fast and f.arc_ratio < t.flat_arc:
            return "drive", 0.75
        if f.arc_ratio >= t.mid_arc and f.max_speed < t.fast:
            return "drop", 0.7   # arcing, but not a hard drive
        if f.max_speed >= t.moderate and f.arc_ratio < t.flat_arc:
            return "volley", 0.6
        return "unknown", 0.4


def classify_shot(stats: TrajectoryStats, model: Optional[Any] = None) -> Dict[str, Any]:
    """Convenience: classify one shot's trajectory."""
    return ShotClassifier(model=model).classify(stats)
