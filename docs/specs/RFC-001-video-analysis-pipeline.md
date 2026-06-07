# RFC-001: Video Analysis Pipeline & Product Backend

| | |
|---|---|
| **Status** | Draft |
| **Author** | Sathish Kumar |
| **Reviewers** | TBD (eng lead, ML lead) |
| **Created / Updated** | 2026-06-07 |
| **Review deadline** | TBD |

## Summary
Design for the backend that turns an uploaded pickleball video into an annotated
output video + coaching report, asynchronously, within a ~2-minute budget. Flask
control plane + GPU vision pipeline (detect → track → fuse → annotate → encode →
coach), on AWS, ~10k videos/mo, rule-coaching MVP first.

## Context & problem statement
Recreational players lack affordable game analysis. We have a working, import-coherent
codebase (detector/tracker/fusion/feedback + async job control plane + auth), all
verified offline against stubs. The open problem is the **real** GPU pipeline,
production reliability, and the pickleball-specific CV (court geometry, actions,
robust tracking). Constraints: ≤2-min latency, single GPU per job, faces → privacy.

## Goals
- One uploaded clip → annotated mp4 + `result.json` via `/jobs/video`, async, ≤~2 min.
- Bounded GPU concurrency (no OOM under load); durable, observable jobs.
- Pickleball-correct analytics: court-relative positions, stable per-player IDs,
  action labels.
- Reuse OSS models/libraries; cloud LLM via a managed provider (Bedrock), not OpenAI.

## Non-Goals
- Real-time/live inference. · Multi-sport. · Training new detectors from scratch
  (use Roboflow pickleball weights). · Mobile clients.

## Proposed design

### Control plane (built)
`POST /jobs/video` (auth) → validate (ffprobe size/duration/codec) → create job →
async worker → poll `GET /jobs/<id>` → download `GET /jobs/<id>/video|result`.
Job state in `JobStore` (→ Redis), outputs in workdir (→ S3 + presigned URLs).

### Vision pipeline (per frame, sampled at FRAME_SKIP)
```
decode(ffmpeg/cv2) → detect(YOLO) → track(ByteTrack) → [pose] → fuse(GameState)
                   → annotate(supervision) → encode(mp4) → coach(rule→cloud)
```
- **Detection:** Ultralytics YOLO with **Roboflow pickleball weights** (PB/paddle/
  ball/player). Swappable via `Config.DETECTOR_MODEL`.
- **Tracking:** supervision **ByteTrack** for stable IDs. Known limitation: ID
  switches under occlusion/crossing (IoU-based); mitigations below.
- **Court geometry:** detect court **keypoints** → compute a **3×3 homography H**
  mapping image points to a top-down "mini-court", giving positions in real court
  units. Standard approach is YOLO/ResNet50 keypoint detection + homography (or
  Hough-line + homography). Enables positioning/zone (kitchen) coaching.
- **Action recognition:** temporal classifier over pose+ball windows per player
  (serve/dink/drive/volley/lob). Framed as **Action Spotting / Precise Event
  Spotting**; 2D/3D/two-stream/skeleton models apply. v1 = heuristic stub
  (`GameStateBuilder._infer_action`), labelled as such.
- **Fusion:** `GameStateBuilder` → players (court coords + side/zone), ball
  (centroid + velocity), action. JSON-serialisable.
- **Coaching:** `rule` (default/fallback) → `cloud` (Bedrock Claude Haiku) later,
  under a wall-clock deadline (P0-5).

### Latency strategy
FRAME_SKIP=3 (~10 fps), GPU batch 16–32, 720p cap, hold-last on skipped frames
(but ball every-frame/interpolated — small fast object), async UX. Target ≤90s/2-min
clip on one g5 GPU.

### AWS topology
ECS Fargate API + ALB · ElastiCache Redis (jobs) · RDS Postgres (users) · Celery on
Redis `gpu` queue · EC2/ECS **g5 Spot** GPU workers (autoscale on queue depth, models
pre-warmed) · S3 (uploads/outputs) + CloudFront · CloudWatch. Bedrock for cloud LLM
(`invoke_model`, Haiku; route Sonnet↔Haiku for cost). S3→SQS→worker is the standard
AWS async-ingest pattern.

## Alternatives considered
- **FastAPI vs Flask** — chose Flask (entry point already Flask-shaped; FastAPI relic
  broken). Revisit if async I/O becomes dominant.
- **Synchronous request processing** — rejected: multi-minute work can't block HTTP;
  async job is mandatory.
- **In-process thread (current) vs Celery** — thread is the single-box bridge;
  Celery+Redis is the durable/scalable target (P1-2). Thread rejected for prod
  (no durability/backpressure/cancel-of-wedged-work).
- **OpenAI direct vs managed cloud** — chose managed (Bedrock/Azure/Vertex) per
  decision: cost/latency/privacy + single-account ops. OpenAI removed (commented).
- **ByteTrack vs DeepSORT/BoT-SORT** — ByteTrack first (simple, fast). For occlusion
  ID-switches, add appearance **ReID** (BoT-SORT / DeepSORT / embeddings) or **EIoU**
  in the 2nd association stage; pickleball's fixed 2/4-player court priors help.
- **Custom tracker/IoU/viz** — rejected: reuse supervision (don't reinvent).

## Trade-offs & risks
- **Seam untested:** control plane only stub-tested; real file handoff / progress
  timing / memory co-fit / encode cost unknown → **M0 must run it** before estimates
  are trusted. (Top risk.)
- **Model co-fit:** YOLO+ByteTrack+pose(+BLIP-2 later) on one GPU — measure memory.
- **Cold starts:** model load not in the 90s budget → pre-warm workers.
- **Ball detection:** small/fast/motion-blur is hard; specialized weights + ball-every-
  frame or TrackNet mitigate.
- **ID switches** scramble per-player coaching → ReID + court priors.
- **LLM variance:** external call in a budget → rule fallback + deadline (built).

## Security / privacy / cost
- Auth + per-user ownership; rate-limit `/auth/*`; token expiry/revocation.
- Faces → retention policy + `DELETE /jobs/<id>` + S3 TTL (GDPR); in-region data.
- Input validation caps abuse/DoS; signed URLs for downloads.
- Cost ≈ $270/mo @10k (rule-only ≈ $120); spot GPU + Haiku + idempotency cache.

## Rollout & testing plan
- **M0** seam test on a GPU box (numbers → `docs/thinking/M0-seam-results.md`).
- pytest suite (offline-verifiable behavior) + CI green; per-stage timing tests.
- Staged AWS deploy (API → Redis/RDS/S3 → Celery/GPU → observability).
- Rollback: feature-flag cloud LLM; rule path always available.

## Open questions
- Retention window (S3 TTL). · IaC: Terraform vs CDK. · Custom auth vs Cognito.
- Action model: train vs off-the-shelf temporal model. · Progress transport (poll→SSE).

## References / Further reading
- RFC/design-doc practice: [Pragmatic Engineer — RFCs & Design Docs](https://blog.pragmaticengineer.com/rfcs-and-design-docs/) · [Fuchsia RFC best practices](https://fuchsia.dev/fuchsia-src/contribute/governance/rfcs/best_practices)
- Court homography: [Roboflow — Camera calibration in sports with keypoints](https://blog.roboflow.com/camera-calibration-sports-computer-vision/) · [HawkEye tennis court keypoints (arXiv 2511.04126)](https://arxiv.org/pdf/2511.04126) · [Pickleball Analytics (Ryan Tolone)](https://ryan-tolone.com/projects/pickleball/)
- Action recognition: [Deep Learning for Sports Video Event Detection (arXiv 2505.03991)](https://arxiv.org/abs/2505.03991) · [Spatiotemporal action recognition I3D+TSN+pose](https://www.sciencedirect.com/science/article/abs/pii/S1746809425008675)
- Tracking / ReID: [Datature — Intro to ByteTrack](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box) · [soccer multi-object tracking (ByteTrack+DeepSORT+BoT-SORT+ReID)](https://github.com/Anudeep007-hub/soccer-multi-object-tracking) · [supervision trackers](https://supervision.roboflow.com/how_to/track_objects/)
- AWS: [Build production-scale inference on Bedrock](https://builder.aws.com/content/30wiaBCETvReEyrrPSROi0FkaUG/build-production-scale-ai-inference-systems-on-amazon-bedrock) · [Bedrock batch inference (S3→SQS→Lambda)](https://aws.amazon.com/blogs/machine-learning/classify-call-center-conversations-with-amazon-bedrock-batch-inference/) · [Invoke Claude on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- Models/weights: [Roboflow — GameChangerv1 pickleball](https://universe.roboflow.com/gamechangerv1/pickleball-detection-1oqlw) · [TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- Internal: `docs/ROADMAP.md`, `docs/DELIVERY_PLAN.md`, `docs/MODELS_AND_REUSE.md`, `docs/BUDGET_PLAN.md`, `docs/assets/system_design.png`
