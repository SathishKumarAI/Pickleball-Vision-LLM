"""Court geometry — image→court homography (pure NumPy, no OpenCV).

Given 4+ correspondences between detected court keypoints (image px) and the known
top-down court model (meters), estimate the 3x3 homography H so player/ball
positions can be expressed in real court coordinates. This unlocks court-aware
analytics (kitchen zone, heatmaps in meters) — the biggest coaching-correctness
upgrade (see docs/specs/RFC-001).

Standard pickleball court (singles/doubles): 6.10 m wide x 13.41 m long; the
non-volley zone ("kitchen") extends 2.13 m from the net on each side.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.41
KITCHEN_DEPTH_M = 2.13
NET_Y_M = COURT_LENGTH_M / 2.0

# Canonical court corners in meters (top-down), order: TL, TR, BR, BL.
COURT_CORNERS_M = np.array([
    [0.0, 0.0],
    [COURT_WIDTH_M, 0.0],
    [COURT_WIDTH_M, COURT_LENGTH_M],
    [0.0, COURT_LENGTH_M],
], dtype=float)


def estimate_homography(src_pts: Sequence[Sequence[float]],
                        dst_pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Estimate H mapping src (image px) -> dst (court m) via the DLT algorithm.

    Args:
        src_pts: >=4 image points [[x, y], ...].
        dst_pts: matching court points [[X, Y], ...].

    Returns:
        3x3 homography matrix (normalized so H[2,2] == 1).
    """
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >=4 matching point pairs")

    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Map image points -> court coordinates. Returns Nx2 array."""
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def in_kitchen(court_xy: Sequence[float]) -> bool:
    """True if a court-coordinate point is inside either non-volley zone."""
    _, y = court_xy
    return abs(y - NET_Y_M) <= KITCHEN_DEPTH_M


def court_side(court_xy: Sequence[float]) -> str:
    """'near' (y > net) or 'far' (y < net) half of the court."""
    return "near" if court_xy[1] >= NET_Y_M else "far"
