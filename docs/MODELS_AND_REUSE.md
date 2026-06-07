# 🔁 Reuse Map — OSS Models & Repos (don't reinvent)

> Curated 2026-06-07. Pretrained models + libraries to plug into each pipeline
> stage instead of writing from scratch. Pick weights here, then set
> `Config.DETECTOR_MODEL` / tracker backend — no new model code needed.

---

## Detection — pretrained pickleball weights (Ultralytics-compatible)

The detector (`src/vision/detection/detector.py`) is plain Ultralytics YOLO, so
any `.pt` weights or Roboflow model drops in via `Config.DETECTOR_MODEL`.

| Model | Classes | Notes |
|-------|---------|-------|
| [Roboflow — GameChangerv1 PickleBall Detection](https://universe.roboflow.com/gamechangerv1/pickleball-detection-1oqlw) | PB, Paddle, ball, player | YOLOv8, 338 imgs — closest to our 4-class need |
| [Roboflow — ak-zcxgt Pickleball](https://universe.roboflow.com/ak-zcxgt/pickleball-uninu-suhi2/model/1) | pickleball | YOLOv11, mAP 65.4 / P 84.2 / R 65.0 |
| [Roboflow — Liberin pickleball-vision](https://universe.roboflow.com/liberin-technologies/pickleball-vision) | court/ball/player | Hosted inference API available |

**Plug in (local weights):** download the `.pt`, set `DETECTOR_MODEL=/path/best.pt`.

**Plug in (Roboflow hosted/edge):** `pip install -e ".[vision]"` (pulls
`inference`), then use Roboflow's `inference` SDK to fetch predictions and map to
our detection schema (`bbox/confidence/class_id/class_name`). A thin
`RoboflowDetector` adapter can wrap this — interface-compatible with
`ObjectDetector.detect()`.

## Ball trajectory — TrackNet family (small/fast ball)

IoU/nearest tracking struggles with a fast pickleball. TrackNet outputs a
heatmap per frame — far better for the ball specifically.

| Repo | Use |
|------|-----|
| [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) | SOTA shuttlecock/ball tracking + trajectory rectification |
| [yastrebksv/TrackNet (PyTorch)](https://github.com/yastrebksv/TrackNet) | Clean impl **with pretrained weights** |
| [hudsong.dev — TrackNetV2 pickleball transfer](https://www.hudsong.dev/pickleball) | Worked example of transfer-learning TrackNet to pickleball |

Integrate as an optional ball backend feeding `GameStateBuilder` (ball centroid).

## Multi-object tracking — Roboflow Supervision (wired ✅)

| Lib | Use |
|-----|-----|
| [roboflow/supervision](https://supervision.roboflow.com/how_to/track_objects/) | **ByteTrack** stable track IDs, `DetectionsSmoother`, annotators, polygon zones |

Already wired: `src/vision/tracking/tracker.py` → `get_tracker("supervision")`
returns a ByteTrack-backed tracker (lazy import; `simple` is the zero-dep
fallback). Set the Pipeline `tracker_backend="supervision"`.

## Court / sports utilities & reference pipelines

| Repo | Use |
|------|-----|
| [roboflow/sports](https://github.com/roboflow/sports) | Court keypoints, perspective transform, sports CV recipes |
| [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) | Full court+ball+player analysis to mirror for fusion |
| [vinod-polinati/pickleball-rally-detection](https://github.com/vinod-polinati/pickleball-rally-detection) | YOLOv8 rally segmentation (physics-based cuts) — pattern for action labels |
| [kpp91302/Pickleball-Analytics](https://github.com/kpp91302/Pickleball-Analytics) | Heatmaps, rally tempo, kitchen-zone usage — analytics ideas |

## Captioning / LLM (already OSS)

- BLIP-2 / LLaVA via `lavis` + HuggingFace `transformers` (in `[llm]`); OpenAI as
  hosted option. See `src/llm/clip_integration.py`, `generate_feedback.py`.

---

### Quick decision for the GPU box
1. `pip install -e ".[vision,llm]"`.
2. Set `DETECTOR_MODEL` to a Roboflow pickleball weight (GameChangerv1 first).
3. `Pipeline(tracker_backend="supervision")`.
4. Optionally add TrackNet for the ball channel.
