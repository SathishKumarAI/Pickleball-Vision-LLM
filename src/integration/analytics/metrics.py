"""Match analytics computed from the per-frame game states.

Pure-Python/NumPy, no GPU — turns the ``states`` list (from
``GameStateBuilder``/``Pipeline.analyze_detections``) into the metrics the product
surfaces: action/shot breakdown, rally tempo, player position heatmap, and
(when a court homography is supplied) kitchen-zone usage in real coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.integration.analytics.court import apply_homography, in_kitchen


def _player_centroids(states: Sequence[Dict[str, Any]]) -> List[List[float]]:
    pts: List[List[float]] = []
    for s in states:
        for p in s.get("players", []) or []:
            c = p.get("centroid")
            if c:
                pts.append([float(c[0]), float(c[1])])
    return pts


def action_breakdown(states: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in states:
        a = s.get("action", "unknown")
        out[a] = out.get(a, 0) + 1
    return out


def rally_tempo(states: Sequence[Dict[str, Any]], fps: float = 30.0) -> Dict[str, float]:
    """Crude rally tempo: shots/sec proxy from action transitions + mean ball speed."""
    actions = [s.get("action") for s in states]
    transitions = sum(1 for a, b in zip(actions, actions[1:]) if a != b)
    duration_s = (len(states) / fps) if fps else 0.0
    speeds = []
    for s in states:
        v = (s.get("ball") or {}).get("velocity")
        if v:
            speeds.append((v[0] ** 2 + v[1] ** 2) ** 0.5)
    return {
        "transitions_per_sec": round(transitions / duration_s, 3) if duration_s else 0.0,
        "mean_ball_speed_px": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "duration_s": round(duration_s, 2),
    }


def position_heatmap(states: Sequence[Dict[str, Any]], bins: int = 8,
                     frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> Dict[str, Any]:
    """2D histogram of player centroids (image space). Returns a bins x bins grid."""
    pts = _player_centroids(states)
    if not pts:
        return {"bins": bins, "grid": [[0] * bins for _ in range(bins)]}
    arr = np.array(pts)
    w = frame_w or float(arr[:, 0].max() or 1)
    h = frame_h or float(arr[:, 1].max() or 1)
    grid, _, _ = np.histogram2d(arr[:, 1], arr[:, 0], bins=bins,
                                range=[[0, h], [0, w]])
    return {"bins": bins, "grid": grid.astype(int).tolist()}


def kitchen_usage(states: Sequence[Dict[str, Any]], homography: Optional[List[List[float]]]) -> Dict[str, float]:
    """Fraction of player-frames inside the non-volley zone (needs court homography)."""
    if homography is None:
        return {"available": 0.0}  # no court calibration
    H = np.asarray(homography, dtype=float)
    total = inside = 0
    for s in states:
        for p in s.get("players", []) or []:
            c = p.get("centroid")
            if not c:
                continue
            court_xy = apply_homography(H, [c])[0]
            total += 1
            if in_kitchen(court_xy):
                inside += 1
    return {"available": 1.0, "kitchen_fraction": round(inside / total, 3) if total else 0.0}


def compute_match_metrics(states: Sequence[Dict[str, Any]], *, fps: float = 30.0,
                          homography: Optional[List[List[float]]] = None,
                          frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate all analytics into the ``metrics`` payload stored on the analysis."""
    return {
        "actions": action_breakdown(states),
        "rally": rally_tempo(states, fps=fps),
        "heatmap": position_heatmap(states, frame_w=frame_w, frame_h=frame_h),
        "kitchen": kitchen_usage(states, homography),
        "num_frames": len(states),
    }
