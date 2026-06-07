# Thinking — Mini-Features Backlog (2026-06-07)

> Planning only. Decomposes remaining product work into small, shippable
> mini-features. Each: **goal · why · OSS reuse · sub-tasks · acceptance · deps ·
> effort · where it runs** (💻 this machine / 🖥️ GPU box). Grounded in current
> repo state (app runnable, vision import-coherent, fusion+feedback+pipeline with
> `rule` backend, supervision ByteTrack wired, reuse map curated).
> Effort: S ≤ ½ day · M ≈ 1–2 days · L ≈ 3–5 days.

Legend: `[ ]` not started · OSS = reuse-don't-reinvent source.

---

## EPIC A — Detection (real pickleball weights)

### A1. Roboflow pickleball detector adapter  · M · 🖥️
- **Goal:** `RoboflowDetector` interchangeable with `ObjectDetector.detect()`, returning the same `{bbox,confidence,class_id,class_name}` schema.
- **Why:** Use pretrained pickleball weights (PB/paddle/ball/player) instead of generic COCO YOLO → real class coverage.
- **OSS:** Roboflow `inference` SDK; weights = GameChangerv1 / ak-zcxgt (see MODELS_AND_REUSE.md).
- **Sub-tasks:**
  - [ ] `src/vision/detection/roboflow_detector.py` — wrap `inference` model load (model id + API key from `Config`/env).
  - [ ] Map Roboflow prediction fields → our detection dict.
  - [ ] `get_detector(backend="ultralytics"|"roboflow")` factory (mirror `get_tracker`).
  - [ ] `Config` fields: `DETECTOR_BACKEND`, `ROBOFLOW_MODEL_ID`, `ROBOFLOW_API_KEY`.
  - [ ] Smoke: 1 frame → ≥0 detections, schema asserted.
- **Acceptance:** `Pipeline(detector_backend="roboflow").process_video(clip)` returns player+ball detections.
- **Deps:** `[vision]` extras + Roboflow key.

### A2. Detector weight config + auto-download  · S · 🖥️
- **Goal:** One config knob to pick local `.pt` vs named Ultralytics weight vs Roboflow.
- **Sub-tasks:**
  - [ ] Document `DETECTOR_MODEL` precedence (already partly in loader).
  - [ ] Cache downloaded weights under `Config.MODEL_DIR`.
  - [ ] Fail-fast clear error when weight missing.
- **Acceptance:** Switching weight = 1 env var, no code change.

---

## EPIC B — Tracking & trajectory

### B1. TrackNet ball-trajectory backend  · L · 🖥️
- **Goal:** Heatmap-based ball tracker for the fast/small pickleball, feeding ball centroid to `GameStateBuilder`.
- **Why:** IoU/nearest (`BallTracker`) loses a fast ball; TrackNet is purpose-built.
- **OSS:** qaz812345/TrackNetV3 or yastrebksv/TrackNet (pretrained); hudsong.dev pickleball transfer recipe.
- **Sub-tasks:**
  - [ ] Vendor/submodule TrackNet inference (3-frame window → heatmap → centroid).
  - [ ] `src/vision/tracking/tracknet_ball.py` adapter → `{centroid,confidence}`.
  - [ ] Wire as optional ball channel in `Pipeline` (separate from supervision player tracking).
  - [ ] Evaluate vs `simple` on the sample clip (hit rate).
- **Acceptance:** Ball centroid populated on ≥80% of in-play frames in sample.
- **Deps:** torch + pretrained TrackNet weights.

### B2. Supervision annotated output video  · M · 🖥️
- **Goal:** Write an annotated `.mp4` (boxes, track IDs, traces) for review/demo.
- **OSS:** `supervision` `BoxAnnotator`/`LabelAnnotator`/`TraceAnnotator` (replace custom `draw_detections`).
- **Sub-tasks:**
  - [ ] `src/vision/visualization/annotate.py` using supervision annotators.
  - [ ] `Pipeline` option `save_annotated=path`.
  - [ ] Retire custom `detector.draw_detections` (or delegate to supervision).
- **Acceptance:** Annotated video with stable IDs produced from a clip.

---

## EPIC C — Scene understanding

### C1. MediaPipe pose integration  · M · 🖥️
- **Goal:** Per-player keypoints → posture/balance feedback.
- **OSS:** MediaPipe Pose (already a dep); `pose_extractor.py` exists.
- **Sub-tasks:**
  - [ ] Run pose on each player crop from detections.
  - [ ] Add `keypoints` to `PlayerState` in game_state.
  - [ ] Feed to `PromptTemplates["pose_feedback"]`.
  - [ ] Compute PCK vs any labelled frames (`metrics.compute_pose_estimation_metrics`).
- **Acceptance:** game_state carries keypoints; pose feedback string generated.

### C2. Court detection / homography  · L · 🖥️
- **Goal:** Map pixels → court coordinates (meters) for zones (kitchen/baseline).
- **OSS:** roboflow/sports court-keypoints, TennisProject court module.
- **Sub-tasks:**
  - [ ] Detect court keypoints/lines.
  - [ ] Estimate homography (cv2.findHomography).
  - [ ] `Config.COURT_*` already has dims — project players/ball to court frame.
  - [ ] Derive `side`/zone from court coords (replace pixel-midline heuristic).
- **Acceptance:** Player positions reported in court meters + zone label.

### C3. Action/shot classifier (replace heuristic)  · L · 🖥️
- **Goal:** Learned shot/action label (serve/volley/dink/drive/lob) vs current speed heuristic.
- **OSS:** VideoMAE (`run_videomae.py` stub) or temporal model on pose+ball features; reference vinod-polinati rally cuts.
- **Sub-tasks:**
  - [ ] Define label set (`Config.SHOT_TYPES` exists).
  - [ ] Feature window builder (ball traj + pose over N frames).
  - [ ] Train/fine-tune small classifier; log to MLflow.
  - [ ] Swap `GameStateBuilder._infer_action` to model when available (keep heuristic fallback).
- **Acceptance:** Action accuracy beats heuristic on a labelled set.

### C4. Rally segmentation  · M · 🖥️
- **Goal:** Split a match video into rally clips.
- **OSS:** vinod-polinati/pickleball-rally-detection (physics-based cuts).
- **Sub-tasks:**
  - [ ] Detect rally start/end from ball motion + gaps.
  - [ ] Emit clip boundaries (timestamps) + optional clip export.
  - [ ] Expose `/analyze/rallies` endpoint.
- **Acceptance:** Sample match → list of rally segments.

---

## EPIC D — LLM / feedback (make it real)

### D1. BLIP-2 / LLaVA caption backend  · M · 🖥️
- **Goal:** Real frame captions feeding `clip_interpretation`.
- **OSS:** `lavis` BLIP-2 / video-LLaVA (`clip_integration.py`, `video_llava_connector.py` exist).
- **Sub-tasks:**
  - [ ] Wire `LLMClipIntegration.generate_frame_captions` to lavis model (lazy).
  - [ ] Caption key frames per rally; attach to game_state.
  - [ ] Cache captions (frame hash) — reuse existing imagehash dedup.
- **Acceptance:** Non-stub captions on sample frames.

### D2. Real coaching LLM backend wired + tested  · S · 🖥️/💻(mock)
- **Goal:** Verify `openai`/`hf` backends in `CoachingFeedbackGenerator` end to end.
- **Sub-tasks:**
  - [ ] Integration test with a mocked client (💻 verifiable here).
  - [ ] Live smoke with `OPENAI_API_KEY` / a small HF model (🖥️).
  - [ ] Prompt tuning for pickleball-specific advice.
- **Acceptance:** Coaching text from a real model for a game_state.

### D3. Feedback quality eval  · M · 🖥️
- **Goal:** Score feedback usefulness (LLM-judge or rubric).
- **Sub-tasks:**
  - [ ] Rubric + small gold set.
  - [ ] LLM-judge harness; log scores to MLflow.
- **Acceptance:** Repeatable feedback score per backend.

---

## EPIC E — Data & ingestion

### E1. YouTube ingestion CLI  · S · 💻(code)/🖥️(run)
- **Goal:** `scripts` entry: URL → downloaded video → sampled frames.
- **OSS:** yt-dlp / pytube (deps present), OpenCV frame sampler (`frame_sampler.py`).
- **Sub-tasks:**
  - [ ] CLI `python -m src... ingest <url>` → `data/raw_videos` + `data/frames`.
  - [ ] Adaptive sampling already in Config (`ADAPTIVE_SAMPLING`).
  - [ ] Log metadata CSV (exists: `video_metadata.csv`).
- **Acceptance:** One command pulls a clip and emits frames.

### E2. Annotation workflow  · M · 💻/🖥️
- **Goal:** Label frames for detection/action training.
- **OSS:** Label Studio / CVAT; Roboflow upload.
- **Sub-tasks:**
  - [ ] Export sampled frames in Label-Studio format.
  - [ ] Round-trip labels → dataset under `data/processed`.
  - [ ] DVC-track datasets.
- **Acceptance:** Labelled set versioned + ready for training.

---

## EPIC F — Serving, API, UX

### F1. Async video job API  · M · 💻
- **Goal:** `/analyze/video` returns a job id; poll for result (long videos).
- **Sub-tasks:**
  - [ ] In-proc job store + status endpoint (no broker first).
  - [ ] Stream progress (frames processed).
  - [ ] Result schema versioned.
- **Acceptance:** Upload → job id → poll → result. (Mockable 💻.)

### F2. Web dashboard  · L · 💻
- **Goal:** Upload clip, see annotated video + game-states + feedback + heatmaps.
- **OSS:** Streamlit (fast) or the existing Svelte shell.
- **Sub-tasks:**
  - [ ] Streamlit page hitting `/analyze`.
  - [ ] Render states timeline + feedback.
  - [ ] Embed annotated video + court heatmap.
- **Acceptance:** End-to-end demo from browser.

### F3. Analytics: heatmaps & rally tempo  · M · 💻
- **Goal:** Player position heatmap, kitchen-zone usage, rally tempo.
- **OSS:** kpp91302/Pickleball-Analytics ideas; matplotlib.
- **Sub-tasks:**
  - [ ] Aggregate centroids over a match → heatmap.
  - [ ] Zone occupancy from court coords (needs C2).
  - [ ] Expose `/analyze/analytics`.
- **Acceptance:** Heatmap PNG + tempo numbers from a match.

---

## EPIC G — Platform / quality

### G1. Test suite revival + CI green  · M · 💻
- **Goal:** `pytest` runs; CI on push.
- **Sub-tasks:**
  - [ ] Rewire skipped `tests/test_collection.py` to current modules (or delete).
  - [ ] Unit tests: game_state actions, feedback rule, tracker predict, /analyze.
  - [ ] Fix `.github/workflows` to install `[dev]` + run pytest + ruff.
- **Acceptance:** Green CI badge; ≥ the offline-verifiable paths covered.

### G2. Config consolidation  · S · 💻
- **Goal:** One canonical config story (`core.config`), drop `fusion/config` env-loader overlap.
- **Sub-tasks:**
  - [ ] Merge `fusion/config/__init__.load_config` into `core.config`.
  - [ ] Document env profiles (dev/prod `.env`).
- **Acceptance:** Single import path for all config.

### G3. MLflow tracking integration  · M · 🖥️
- **Goal:** Log detection/tracking/LLM runs (params, metrics, artifacts).
- **OSS:** MLflow (dep present); `run_mlflow_experiment.py` exists.
- **Sub-tasks:**
  - [ ] `Config.USE_MLFLOW` switch in Pipeline.
  - [ ] Log per-run metrics from `metrics.py`.
  - [ ] Artifacts: annotated video, game_state json.
- **Acceptance:** Runs visible in MLflow UI.

### G4. Docker serving validation  · M · 🖥️
- **Goal:** `deployment/` Docker actually builds + serves the Flask app + pipeline.
- **Sub-tasks:**
  - [ ] Update Dockerfile to install `.[vision,llm]`.
  - [ ] Compose: app + (optional) MLflow + Prometheus.
  - [ ] Healthcheck hits `/health`.
- **Acceptance:** `docker compose up` → working `/analyze`.

---

## Suggested sequencing
1. **Verifiable-here first (💻):** G1 tests/CI → D2 mock LLM test → F1 job API → F3 analytics scaffolds.
2. **GPU box, high value:** A1 Roboflow detector → B2 annotated video → C1 pose → D1 captions.
3. **Then heavier:** C2 court homography → C3 action classifier → B1 TrackNet → F2 dashboard.
4. **Platform:** G3 MLflow, G4 Docker alongside.

## Notes
- Keep the lazy-import + dep-free-fallback pattern for every new vision/LLM module so the offline `/analyze` path stays green and code stays parse-verifiable here.
- Each mini-feature = its own branch/commit + a `docs/thinking/` iteration note (per the working agreement).
- Reuse first (MODELS_AND_REUSE.md) before writing any model code.
