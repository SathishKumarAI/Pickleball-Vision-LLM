# Thinking — OSS reuse execution plan (2026-06-07)

## Goal
Don't reinvent. Wire the project to existing OSS models/repos. Implement on this
(no-ML-deps) machine; runtime verification happens on the GPU box.

## Research findings → what to reuse

| Pipeline stage | Reuse (OSS) | Why / how to plug in |
|---|---|---|
| Ball/player detection | **Roboflow Universe pickleball weights** (GameChangerv1: PB/Paddle/ball/player YOLOv8; ak-zcxgt YOLOv11; Liberin pickleball-vision) | Already Ultralytics YOLO — swap via `Config.DETECTOR_MODEL` or a Roboflow `inference` backend. No new detector code. |
| Ball trajectory | **TrackNetV3** (qaz812345) / **yastrebksv/TrackNet** (pretrained); pickleball transfer shown at hudsong.dev | Heatmap-based small-fast-ball tracker — better than IoU nearest. Optional backend. |
| Multi-object tracking | **roboflow/supervision** ByteTrack + annotators | Replace hand-rolled `BallTracker`/viz. Lazy import, fallback to current. |
| Sports CV utils / court | **roboflow/sports**, **yastrebksv/TennisProject** | Court keypoints, perspective — reference for fusion. |
| Pickleball pipelines (reference) | vinod-polinati/pickleball-rally-detection, kpp91302/Pickleball-Analytics | Rally segmentation, heatmaps — patterns to copy, not deps. |

## Decisions
1. **Detection** stays Ultralytics (already OSS). Make the *weights* swappable +
   add an optional Roboflow `inference` backend. Document the pickleball weights.
2. **Tracking**: introduce `supervision.ByteTrack`-backed tracker as the preferred
   backend; keep the existing simple tracker as a zero-dep fallback. Lazy import.
3. **Annotation/viz**: prefer `supervision` annotators over custom drawing (TODO,
   lower priority — current draw works).
4. Add `supervision` + `inference` to `pyproject [vision]` extras.
5. Ship a curated `docs/MODELS_AND_REUSE.md` so the GPU-box operator can pick
   weights without re-researching.

## Constraints
- No ML deps here → all new integrations use **lazy imports + graceful fallback**,
  verified by parse + dep-free smoke. Real inference = GPU box.
- Keep changes additive; don't break the green `/analyze` dep-free path.

## Iteration order (this session)
1. This note. ✅
2. `pyproject.toml`: add supervision + inference to [vision].
3. `src/vision/tracking/tracker.py`: add `SupervisionTracker` (ByteTrack, lazy) +
   keep `BallTracker`; factory `get_tracker(backend=...)`.
4. `docs/MODELS_AND_REUSE.md`: curated table + plug-in snippets.
5. Verify (parse + dep-free smoke) → commit → push.
6. Next note documents what changed + what's left for the GPU box.
