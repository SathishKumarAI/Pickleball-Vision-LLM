"""Async video-analysis job API.

The product control plane: upload a game video, get a job id back immediately,
poll for progress, then download the annotated output video + insights.

* ``POST /jobs/video``    multipart ``video`` → ``{job_id}`` (202).
* ``GET  /jobs/<id>``     job status + progress.
* ``GET  /jobs/<id>/result``  insights JSON (when done).
* ``GET  /jobs/<id>/video``   annotated output mp4 download (when done).

The heavy work runs in a background thread via :data:`src.services.jobs.store`.
The processor lazy-imports the vision Pipeline, so this module imports without
the ``[vision]`` extras; the control-plane lifecycle is testable with a stub.
"""

from __future__ import annotations

import os
import tempfile

from flask import Blueprint, g, jsonify, request, send_file

from src.api.blueprints.auth import token_required
from src.services.jobs import store

bp = Blueprint("jobs", __name__)


def _owned_or_error(job_id: str):
    """Return (job, None) if the current user owns the job, else (None, response)."""
    job = store.get(job_id)
    if job is None:
        return None, (jsonify(error="job not found"), 404)
    if job.meta.get("user_id") != g.user["id"]:
        return None, (jsonify(error="forbidden"), 403)
    return job, None


def _build_processor(video_path: str, backend: str, tracker: str, annotate_path: str):
    """Return a Processor closure that runs the Pipeline with progress updates."""
    def processor(job, progress_cb):
        progress_cb(0.02, "loading pipeline")
        from src.pipeline import Pipeline  # lazy: needs [vision] at run time
        pipe = Pipeline(feedback_backend=backend, tracker_backend=tracker)
        result = pipe.process_video(
            video_path,
            annotate_path=annotate_path,
            progress_cb=lambda f, m="processing": progress_cb(0.05 + 0.9 * f, m),
        )
        progress_cb(0.98, "finalizing")
        job.meta["video_path"] = annotate_path
        return result
    return processor


@bp.post("/jobs/video")
@token_required
def submit_video():
    """Accept a video upload and start an async analysis job (auth required)."""
    if "video" not in request.files:
        return jsonify(error="missing 'video' file"), 400
    f = request.files["video"]
    backend = request.form.get("backend", "rule")
    tracker = request.form.get("tracker", "supervision")

    workdir = tempfile.mkdtemp(prefix="pvjob_")
    in_path = os.path.join(workdir, f.filename or "input.mp4")
    out_path = os.path.join(workdir, "annotated.mp4")
    f.save(in_path)

    job = store.create(meta={"input": in_path, "workdir": workdir, "user_id": g.user["id"]})
    store.submit(job, _build_processor(in_path, backend, tracker, out_path))
    return jsonify(job_id=job.id, status=job.status.value), 202


@bp.get("/jobs/<job_id>")
@token_required
def job_status(job_id: str):
    job, err = _owned_or_error(job_id)
    if err:
        return err
    return jsonify(job.to_dict())


@bp.get("/jobs/<job_id>/result")
@token_required
def job_result(job_id: str):
    job, err = _owned_or_error(job_id)
    if err:
        return err
    if job.status.value != "done":
        return jsonify(error="not ready", status=job.status.value), 409
    return jsonify(job.result)


@bp.get("/jobs/<job_id>/video")
@token_required
def job_video(job_id: str):
    job, err = _owned_or_error(job_id)
    if err:
        return err
    path = job.meta.get("video_path")
    if job.status.value != "done" or not path or not os.path.exists(path):
        return jsonify(error="annotated video not ready", status=job.status.value), 409
    return send_file(path, mimetype="video/mp4", as_attachment=True,
                     download_name=f"annotated_{job_id}.mp4")
