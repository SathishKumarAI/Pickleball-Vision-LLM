"""Analysis API blueprint.

Reconstructs the detection/analysis endpoint that lived in the removed
``src/temp/api`` (FastAPI relic), now as a Flask blueprint wired to the real
:class:`src.pipeline.Pipeline`.

* ``POST /analyze`` with JSON ``{"detections": [[...], ...], "fps": 30}`` runs the
  dependency-free fusion + feedback path (no model needed) — handy for testing
  and for clients that already have detections.
* ``POST /analyze/video`` with form-file ``video`` runs the full vision pipeline;
  returns 503 if the ``[vision]`` extras aren't installed.
"""

from __future__ import annotations

import os
import tempfile

from flask import Blueprint, jsonify, request

bp = Blueprint("analyze", __name__)


@bp.post("/analyze")
def analyze_detections():
    """Fuse pre-computed detections into game state + coaching feedback."""
    from src.pipeline import Pipeline

    payload = request.get_json(silent=True) or {}
    detections = payload.get("detections")
    if detections is None:
        return jsonify(error="missing 'detections' (list of per-frame detection lists)"), 400
    fps = float(payload.get("fps", 30.0))
    frame_height = payload.get("frame_height")
    backend = payload.get("backend", "rule")

    pipe = Pipeline(feedback_backend=backend, frame_height=frame_height)
    result = pipe.analyze_detections(detections, fps=fps, frame_height=frame_height)
    return jsonify(result)


@bp.post("/analyze/video")
def analyze_video():
    """Run the full vision pipeline on an uploaded video (needs [vision] extras)."""
    if "video" not in request.files:
        return jsonify(error="missing 'video' file"), 400
    f = request.files["video"]
    backend = request.form.get("backend", "rule")

    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(f.filename)[1], delete=False)
    try:
        f.save(tmp.name)
        tmp.close()
        from src.pipeline import Pipeline
        try:
            result = Pipeline(feedback_backend=backend).process_video(tmp.name)
        except ImportError as e:
            return jsonify(error=f"vision extras not installed: {e}"), 503
        return jsonify(result)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
