"""Ball trajectory analysis (pure NumPy).

Turns a sequence of ball centroids (some possibly missing) into kinematics the
rest of the system can reason about: a gap-filled smoothed path, per-frame
velocity/acceleration/speed, bounce events (ground contacts), and the trajectory's
vertical arc. Image coordinates assume **y grows downward** (OpenCV convention),
so a "bounce" is a local **maximum** in y where vertical velocity flips sign.

No torch/cv2 — this runs and tests offline. Feed it the ball centroids produced by
``GameStateBuilder`` / the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Optional[Sequence[float]]  # [x, y] or None when the ball wasn't detected


@dataclass
class Bounce:
    frame: int
    position: Tuple[float, float]


@dataclass
class TrajectoryStats:
    n_frames: int
    n_detected: int
    path: np.ndarray          # (N, 2) gap-filled + smoothed centroids
    velocity: np.ndarray      # (N, 2) px/frame
    speed: np.ndarray         # (N,) px/frame
    bounces: List[Bounce] = field(default_factory=list)
    peak_frame: Optional[int] = None   # highest point of flight (min y)
    max_speed: float = 0.0
    mean_speed: float = 0.0
    apex_height_px: float = 0.0        # vertical span of the arc
    n_rejected: int = 0                # detection spikes removed as outliers

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "n_detected": self.n_detected,
            "n_rejected": self.n_rejected,
            "bounces": [{"frame": b.frame, "position": list(b.position)} for b in self.bounces],
            "peak_frame": self.peak_frame,
            "max_speed_px": round(self.max_speed, 2),
            "mean_speed_px": round(self.mean_speed, 2),
            "apex_height_px": round(self.apex_height_px, 2),
        }


def _interpolate_gaps(xs: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs in a 1-D array; hold ends."""
    idx = np.arange(len(xs))
    good = ~np.isnan(xs)
    if good.sum() == 0:
        return np.zeros_like(xs)
    if good.sum() == 1:
        return np.full_like(xs, xs[good][0])
    return np.interp(idx, idx[good], xs[good])


def _median_filter(a: np.ndarray, w: int = 5) -> np.ndarray:
    """Median filter (prefer scipy; numpy sliding fallback)."""
    try:
        from scipy.ndimage import median_filter  # OSS engine
        return median_filter(a, size=w, mode="nearest")
    except ImportError:
        half = w // 2
        out = a.copy()
        for i in range(len(a)):
            lo, hi = max(0, i - half), min(len(a), i + half + 1)
            out[i] = np.median(a[lo:hi])
        return out


def _reject_outliers(x: np.ndarray, y: np.ndarray, k: float = 2.5,
                     max_jump_px: Optional[float] = None,
                     min_spike: float = 15.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Replace **isolated detection spikes** with the neighbour midpoint.

    A bad detection (jersey/logo/line false positive) is a point that jumps far away
    then returns the next frame: both edges to it are large, but the "bridge" between
    its neighbours is small. A real apex/bounce is the opposite — gradual, with the
    bridge ≈ sum of edges — so this leaves true extrema untouched (unlike a plain
    median filter, which flattens them).

    Flag i when ``min(jump_in, jump_out) > k·bridge + min_spike`` (or > ``max_jump_px``).
    Returns (x, y, n_rejected).
    """
    n = len(x)
    if n < 3:
        return x, y, 0
    rejected = 0
    for i in range(1, n - 1):
        jin = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        jout = float(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]))
        bridge = float(np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]))
        edge = min(jin, jout)
        is_spike = edge > k * bridge + min_spike
        if max_jump_px is not None:
            is_spike = is_spike or edge > max_jump_px
        if is_spike:
            x[i] = (x[i - 1] + x[i + 1]) / 2.0
            y[i] = (y[i - 1] + y[i + 1]) / 2.0
            rejected += 1
    return x, y, rejected


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    """De-noise a 1-D signal. Prefer scipy's Savitzky-Golay (preserves peaks much
    better than a box filter — matters for bounce/apex detection); fall back to a
    moving average if scipy isn't installed."""
    if w <= 1 or len(a) < w:
        return a
    try:
        from scipy.signal import savgol_filter  # OSS engine
        win = w if w % 2 == 1 else w + 1          # savgol needs an odd window
        win = min(win, len(a) if len(a) % 2 == 1 else len(a) - 1)
        if win >= 3:
            return savgol_filter(a, win, polyorder=min(2, win - 1))
    except ImportError:
        pass
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="same")


def analyze_trajectory(centroids: Sequence[Point], smooth_window: int = 3,
                       min_bounce_speed: float = 1.0, reject_outliers: bool = True,
                       outlier_k: float = 4.0,
                       max_jump_px: Optional[float] = None) -> TrajectoryStats:
    """Compute trajectory kinematics + bounces from ball centroids.

    Args:
        centroids: per-frame [x, y] or None.
        smooth_window: window for de-noising the path.
        min_bounce_speed: ignore direction flips slower than this (jitter guard).
        reject_outliers: replace detection spikes with a local median (robust to a
            noisy/imperfect detector — see _reject_outliers).
        outlier_k: MAD multiplier for the outlier threshold.
        max_jump_px: optional hard cap on plausible per-frame displacement.
    """
    n = len(centroids)
    raw = np.array([[np.nan, np.nan] if c is None else [float(c[0]), float(c[1])]
                    for c in centroids], dtype=float) if n else np.zeros((0, 2))
    n_detected = int(np.sum(~np.isnan(raw[:, 0]))) if n else 0

    if n == 0 or n_detected == 0:
        return TrajectoryStats(n_frames=n, n_detected=0,
                               path=np.zeros((n, 2)), velocity=np.zeros((n, 2)),
                               speed=np.zeros(n))

    x = _interpolate_gaps(raw[:, 0])
    y = _interpolate_gaps(raw[:, 1])
    n_rejected = 0
    if reject_outliers:
        x, y, n_rejected = _reject_outliers(x, y, k=outlier_k, max_jump_px=max_jump_px)
    x = _smooth(x, smooth_window)
    y = _smooth(y, smooth_window)
    path = np.column_stack([x, y])

    velocity = np.zeros_like(path)
    if n >= 2:
        velocity[1:] = np.diff(path, axis=0)
        velocity[0] = velocity[1]
    speed = np.linalg.norm(velocity, axis=1)

    # Gap-aware speed: a velocity is real only when *both* endpoints were actually
    # detected. Velocity across an interpolated gap is an artefact of filling, not
    # ball motion — excluding it stops sparse detections inflating max/mean speed.
    detected = ~np.isnan(raw[:, 0])
    valid_vel = detected.copy()
    valid_vel[1:] &= detected[:-1]
    valid_vel[0] = False
    real_speed = speed[valid_vel]

    # Bounces: vertical velocity flips from down (+) to up (-) => local max in y.
    vy = velocity[:, 1]
    bounces: List[Bounce] = []
    for i in range(1, n - 1):
        if vy[i - 1] > min_bounce_speed and vy[i + 1] < -min_bounce_speed:
            bounces.append(Bounce(frame=i, position=(float(path[i, 0]), float(path[i, 1]))))

    peak_frame = int(np.argmin(y))  # smallest y == highest in image
    return TrajectoryStats(
        n_frames=n, n_detected=n_detected, path=path, velocity=velocity, speed=speed,
        bounces=bounces, peak_frame=peak_frame,
        max_speed=float(real_speed.max()) if real_speed.size else float(speed.max()),
        mean_speed=float(real_speed.mean()) if real_speed.size else float(speed.mean()),
        apex_height_px=float(y.max() - y.min()), n_rejected=n_rejected,
    )


def ball_centroids_from_states(states: Sequence[dict]) -> List[Point]:
    """Extract the per-frame ball centroid list from game states."""
    out: List[Point] = []
    for s in states:
        ball = s.get("ball") or {}
        out.append(ball.get("centroid"))
    return out
