"""Rally segmentation + match-level shot analysis (pure NumPy).

Ties the trajectory + shot pieces together into the match analytics the product
surfaces (PB-Vision inspired: rallies, longest rally, per-shot type, serve/return,
shot counts). Pure-Python over the game states — runs and tests offline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.vision.analysis.actions import ShotClassifier
from src.vision.analysis.trajectory import (
    Point, analyze_trajectory, ball_centroids_from_states,
)


def segment_rallies(centroids: Sequence[Point], max_gap: int = 12,
                    min_len: int = 4) -> List[Tuple[int, int]]:
    """Group frames where the ball is in play into rallies.

    A rally is a run of ball-present frames; runs separated by more than
    ``max_gap`` missing frames are different rallies. Runs shorter than
    ``min_len`` are dropped (noise).

    Returns inclusive (start_frame, end_frame) ranges.
    """
    present = [c is not None for c in centroids]
    rallies: List[Tuple[int, int]] = []
    start = None
    gap = 0
    for i, p in enumerate(present):
        if p:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > max_gap:
                    rallies.append((start, i - gap))
                    start = None
    if start is not None:
        rallies.append((start, len(present) - 1 - (0 if present[-1] else gap)))
    return [(s, e) for s, e in rallies if e - s + 1 >= min_len]


def split_into_shots(centroids: Sequence[Point]) -> List[Tuple[int, int]]:
    """Split a rally's ball track into shots at bounce/contact points.

    Each bounce (ground contact) ends a shot; segments between bounces are shots.
    Falls back to the whole rally as one shot if no bounce is found.
    """
    stats = analyze_trajectory(centroids)
    cuts = [b.frame for b in stats.bounces]
    bounds = [0] + cuts + [len(centroids) - 1]
    shots: List[Tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a >= 2:
            shots.append((a, b))
    return shots or [(0, len(centroids) - 1)]


def analyze_match(states: Sequence[Dict[str, Any]], fps: float = 30.0,
                  classifier: ShotClassifier | None = None) -> Dict[str, Any]:
    """Full match intelligence: rallies, shots, types, serve/return, counts."""
    clf = classifier or ShotClassifier()
    centroids = ball_centroids_from_states(states)
    rallies = segment_rallies(centroids)

    rally_out: List[Dict[str, Any]] = []
    shot_counts: Dict[str, int] = {}
    all_shots: List[Dict[str, Any]] = []

    for ri, (rs, re) in enumerate(rallies):
        rc = centroids[rs:re + 1]
        shots: List[Dict[str, Any]] = []
        for si, (a, b) in enumerate(split_into_shots(rc)):
            seg = rc[a:b + 1]
            stats = analyze_trajectory(seg)
            res = clf.classify(stats)
            # First two shots of a rally are serve then return.
            if si == 0:
                res["shot"] = "serve"
            elif si == 1 and res["shot"] in ("drive", "drop", "unknown"):
                res["shot"] = "return"
            res["rally"] = ri
            res["start_frame"] = rs + a
            res["end_frame"] = rs + b
            shots.append(res)
            all_shots.append(res)
            shot_counts[res["shot"]] = shot_counts.get(res["shot"], 0) + 1
        rally_out.append({
            "index": ri, "start_frame": rs, "end_frame": re,
            "length_frames": re - rs + 1,
            "length_s": round((re - rs + 1) / fps, 2) if fps else 0.0,
            "shots": len(shots),
        })

    longest = max(rally_out, key=lambda r: r["length_frames"]) if rally_out else None
    return {
        "rallies": rally_out,
        "num_rallies": len(rally_out),
        "longest_rally_s": longest["length_s"] if longest else 0.0,
        "shots": all_shots,
        "shot_counts": shot_counts,
        "total_shots": len(all_shots),
    }
