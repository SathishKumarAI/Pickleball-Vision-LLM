# ADR-0004: Reuse OSS, don't reinvent

**Status:** Accepted · **Date:** 2026-06-07

## Context
The vision pipeline needs detection, tracking, annotation, ball trajectory, and
court geometry. Hand-rolling these (custom trackers, IoU, drawing) is slower and
worse than mature OSS. The user directed: use existing open-source projects.

## Decision
Prefer well-maintained OSS over custom code:
- **Ultralytics YOLO** + **Roboflow Universe** pickleball weights for detection.
- **supervision** ByteTrack + annotators (and **BoT-SORT/ReID** for occlusion).
- **TrackNet** for fast-ball trajectory.
- **MediaPipe** pose; **loguru**, **torchvision** ops where applicable.
Custom code only for glue (game-state fusion, court homography, coaching) where no
drop-in exists. When swapping custom→lib, do it only where verifiable.

## Consequences
- ✅ Less code to maintain; better accuracy; faster delivery.
- ✅ Detection weights swap via config (`DETECTOR_MODEL`) — no model code.
- ➖ More third-party deps (lazy-imported in the Modal image only).

## References
- `docs/MODELS_AND_REUSE.md` · `src/vision/tracking/tracker.py`
