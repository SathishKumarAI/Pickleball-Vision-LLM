"""Offline tests for the vision-intelligence layer (trajectory, shots, rallies).

Pure-NumPy/scipy/sklearn — no GPU. Synthetic trajectories exercise the classifier
taxonomy and the rally/shot segmentation.
"""

import numpy as np

from src.vision.analysis.trajectory import analyze_trajectory
from src.vision.analysis.actions import classify_shot, ShotClassifier
from src.vision.analysis.rally import segment_rallies, split_into_shots, analyze_match


def _line(n, vx, y0=100.0, x0=0.0):
    return [[x0 + vx * i, y0] for i in range(n)]


def _arc(n, vx, height):
    # parabola: y dips "up" (smaller y) by `height` then returns. x advances by vx.
    t = np.linspace(0, np.pi, n)
    return [[vx * i, 100.0 - height * np.sin(t[i])] for i in range(n)]


# -- trajectory --------------------------------------------------------------

def test_trajectory_bounce_and_apex():
    # ball falls (y up) to a peak then rises -> one bounce at the bottom
    y = [100, 110, 120, 130, 120, 110, 100]
    cents = [[float(i), float(v)] for i, v in enumerate(y)]
    s = analyze_trajectory(cents, smooth_window=3)
    assert len(s.bounces) >= 1
    assert s.n_detected == 7 and s.apex_height_px > 0


def test_trajectory_handles_gaps():
    cents = [[0, 100], None, [20, 100], None, [40, 100]]
    s = analyze_trajectory(cents)
    assert s.n_detected == 3 and s.path.shape == (5, 2)


# -- shot classification -----------------------------------------------------

def test_classify_drive():
    s = analyze_trajectory(_line(21, vx=20))          # fast + flat
    assert classify_shot(s)["shot"] == "drive"


def test_classify_lob():
    s = analyze_trajectory(_arc(21, vx=5, height=150)) # big arc
    assert classify_shot(s)["shot"] == "lob"


def test_classify_dink():
    s = analyze_trajectory(_line(15, vx=2))            # slow + short
    assert classify_shot(s)["shot"] == "dink"


def test_classify_drop():
    s = analyze_trajectory(_arc(21, vx=6, height=50))  # soft moderate arc, travels
    assert classify_shot(s)["shot"] == "drop"


def test_sklearn_model_head():
    # a trained model head takes precedence over the rules
    from sklearn.tree import DecisionTreeClassifier
    X = [[20, 20, 400, 0, 0, 0, 0], [2, 3, 30, 5, 0.16, 0, 0]]
    y = ["drive", "dink"]
    clf = ShotClassifier(model=DecisionTreeClassifier().fit(X, y))
    s = analyze_trajectory(_line(21, vx=20))
    assert clf.classify(s)["shot"] in ("drive", "dink")  # model decides


# -- rally / shot segmentation ----------------------------------------------

def test_segment_rallies_splits_on_gap():
    cents = ([[i, 100] for i in range(10)]      # rally 1
             + [None] * 20                       # gap
             + [[i, 100] for i in range(10)])    # rally 2
    rallies = segment_rallies(cents, max_gap=12, min_len=4)
    assert len(rallies) == 2


def test_split_into_shots_on_bounce():
    y = [100, 115, 130, 115, 100, 115, 130]      # one clear bounce mid-track
    cents = [[float(i * 5), float(v)] for i, v in enumerate(y)]
    shots = split_into_shots(cents)
    assert len(shots) >= 2


def test_analyze_match_end_to_end():
    states = []
    for i in range(12):
        states.append({"ball": {"centroid": [i * 8.0, 100.0]}, "players": []})
    states += [{"ball": {}, "players": []} for _ in range(20)]   # gap
    for i in range(12):
        states.append({"ball": {"centroid": [i * 8.0, 120.0]}, "players": []})
    out = analyze_match(states, fps=30)
    assert out["num_rallies"] == 2
    assert out["total_shots"] >= 2
    assert any(s["shot"] == "serve" for s in out["shots"])  # first shot of a rally
    assert out["longest_rally_s"] > 0
