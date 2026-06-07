"""Phase 6 offline tests: court homography + match analytics (pure numpy)."""

import numpy as np

from src.integration.analytics.court import (
    COURT_CORNERS_M, apply_homography, court_side, estimate_homography, in_kitchen,
)
from src.integration.analytics.metrics import (
    action_breakdown, compute_match_metrics, position_heatmap, rally_tempo,
)


def test_homography_maps_corners_to_court():
    # A simple image quad -> known court corners.
    img = [[100, 50], [500, 50], [500, 900], [100, 900]]
    H = estimate_homography(img, COURT_CORNERS_M)
    mapped = apply_homography(H, img)
    assert np.allclose(mapped, COURT_CORNERS_M, atol=1e-6)


def test_homography_interpolates_center():
    img = [[0, 0], [10, 0], [10, 20], [0, 20]]
    dst = [[0, 0], [6.10, 0], [6.10, 13.41], [0, 13.41]]
    H = estimate_homography(img, dst)
    center = apply_homography(H, [[5, 10]])[0]
    assert np.allclose(center, [3.05, 6.705], atol=1e-6)


def test_kitchen_and_side():
    assert in_kitchen([3.0, 6.705]) is True       # at the net
    assert in_kitchen([3.0, 0.5]) is False         # baseline
    assert court_side([3.0, 10.0]) == "near"
    assert court_side([3.0, 2.0]) == "far"


def test_action_breakdown_and_tempo():
    states = [
        {"action": "serve-or-reset", "ball": {"velocity": [1, 1]}, "players": []},
        {"action": "rally", "ball": {"velocity": [10, 0]}, "players": []},
        {"action": "rally", "ball": {"velocity": [10, 0]}, "players": []},
        {"action": "no-ball", "ball": {}, "players": []},
    ]
    assert action_breakdown(states) == {"serve-or-reset": 1, "rally": 2, "no-ball": 1}
    t = rally_tempo(states, fps=4.0)
    assert t["duration_s"] == 1.0 and t["transitions_per_sec"] == 2.0
    assert t["mean_ball_speed_px"] > 0


def test_heatmap_grid_shape():
    states = [{"players": [{"centroid": [50, 60]}, {"centroid": [200, 300]}]}]
    hm = position_heatmap(states, bins=4, frame_w=400, frame_h=400)
    assert hm["bins"] == 4 and len(hm["grid"]) == 4 and len(hm["grid"][0]) == 4
    assert sum(sum(r) for r in hm["grid"]) == 2  # two centroids binned


def test_compute_match_metrics_with_homography():
    states = [{"action": "rally", "ball": {"velocity": [5, 5]},
               "players": [{"centroid": [5, 10]}]}]
    H = estimate_homography([[0, 0], [10, 0], [10, 20], [0, 20]],
                            [[0, 0], [6.10, 0], [6.10, 13.41], [0, 13.41]]).tolist()
    m = compute_match_metrics(states, fps=30, homography=H, frame_h=480)
    assert m["kitchen"]["available"] == 1.0
    assert "kitchen_fraction" in m["kitchen"]
    assert m["actions"] == {"rally": 1} and m["num_frames"] == 1
