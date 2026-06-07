"""Upload validation (P0-2).

Enforce the bounds the latency budget assumes, *before* accepting a job. A
4-hour / 4K upload is an unbounded job and a trivial DoS; malformed containers
are an attack surface against cv2/ffmpeg. Probe with ``ffprobe`` — never trust
the client's content-type or file extension.

Raises :class:`UploadError` with an HTTP status hint; the API maps it to
413/415/422.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict

MAX_BYTES = 300 * 1024 * 1024       # 300 MB
MAX_DURATION_S = 180                 # 2-min clip + margin
MAX_PIXELS = 1920 * 1080            # downscale anything larger on decode
ALLOWED_CODECS = {"h264", "hevc", "vp9", "av1", "mpeg4"}


class UploadError(ValueError):
    """Invalid upload. ``status`` is the HTTP code the API should return."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def ffprobe(path: str) -> Dict[str, Any]:
    """Return ffprobe JSON for a media file, or raise UploadError."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError as e:  # ffprobe not installed
        raise UploadError("server media probe unavailable", status=500) from e
    except subprocess.TimeoutExpired as e:
        raise UploadError("media probe timed out", status=422) from e
    if out.returncode != 0:
        raise UploadError("unreadable / malformed media", status=422)
    return json.loads(out.stdout)


def validate_upload(path: str, size_bytes: int) -> Dict[str, Any]:
    """Validate a video upload. Returns probe metadata for downstream reuse."""
    if size_bytes <= 0:
        raise UploadError("empty file", status=422)
    if size_bytes > MAX_BYTES:
        raise UploadError(f"file too large (> {MAX_BYTES} bytes)", status=413)

    meta = ffprobe(path)
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise UploadError("no video stream", status=422)

    codec = video.get("codec_name")
    if codec not in ALLOWED_CODECS:
        raise UploadError(f"unsupported codec: {codec}", status=415)

    duration = float(meta.get("format", {}).get("duration", 0) or 0)
    if duration <= 0 or duration > MAX_DURATION_S:
        raise UploadError(f"duration {duration:.0f}s out of bounds (max {MAX_DURATION_S}s)",
                          status=422)

    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    meta["_needs_downscale"] = bool(width * height > MAX_PIXELS)
    meta["_duration_s"] = duration
    return meta
