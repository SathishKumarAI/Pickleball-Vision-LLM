# Thinking — OSS reuse, iteration 1 result (2026-06-07)

## Done (verified on this no-ML-deps machine: parse + dep-free smoke green)
- `pyproject [vision]`: added `supervision` (ByteTrack) + `inference` (Roboflow).
- `src/vision/tracking/tracker.py`: rewrote import-safe (`__future__ annotations`,
  lazy numpy/supervision). Now:
  - `BallTracker` (simple, zero-dep) — also gave it a real `predict_next_position`
    (linear extrapolation; verified (5,5)->(10,10) ⇒ (15,15)).
  - `SupervisionTracker` — ByteTrack, lazy import, maps our detection schema ⇄
    `sv.Detections`, returns `tracker_id`.
  - `get_tracker(backend)` factory.
- `Pipeline(tracker_backend=...)` wired to the factory.
- `docs/MODELS_AND_REUSE.md`: curated reuse table (Roboflow pickleball weights,
  TrackNetV3/yastrebksv, supervision, roboflow/sports, TennisProject, reference
  pickleball pipelines) + plug-in steps for the GPU box.

## Why this shape
- Reuse over rewrite: ByteTrack via supervision instead of a hand-rolled tracker;
  detection weights come from Roboflow Universe (detector already Ultralytics, so
  zero new model code). Matches the user's "don't reinvent" directive.
- Lazy imports keep the offline `/analyze` path green and everything parse-checkable
  here; real inference deferred to the GPU box.

## Left for the GPU box (needs extras) — next iterations
1. `RoboflowDetector` adapter (wrap `inference` SDK → our detect() schema) — only
   worth writing/testing with `[vision]` installed.
2. Optional TrackNet ball backend feeding `GameStateBuilder`.
3. Swap custom draw → `supervision` annotators (viz).
4. Run `Pipeline(tracker_backend="supervision").process_video(sample.mp4)` end to
   end; tune `Config.DETECTOR_MODEL` to a Roboflow pickleball weight.

## Note to future self
Don't write the Roboflow/TrackNet adapters blind here — they need live SDK/model
behaviour to verify. Stub the interface only if asked; otherwise implement on the
box where they can run.
