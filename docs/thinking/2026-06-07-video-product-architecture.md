# Thinking — Solution Architecture: "upload game video → annotated output video in minutes"

## Problem
Web product. User uploads a pickleball game video. We return a downloadable
**annotated output video** (boxes, track IDs, ball trail, action labels,
coaching overlay) + a JSON insights report, within a **couple of minutes**.
Must run on our existing codebase (Pipeline / detector / supervision tracker /
fusion / feedback).

## Hard constraint = latency budget (~2 min wall-clock)
A 2-min, 30 fps clip = 3,600 frames. Per-frame YOLO+track+annotate ≈ 30–60 ms on
a modern GPU ⇒ ~2–4 min if we touch every frame. Too slow. Levers:

| Lever | Effect | Default |
|---|---|---|
| **Frame skip** (sample 1/N) | linear speedup, fine for coaching cadence | N=3 (≈10 fps) |
| **GPU batch inference** | amortize model overhead | batch 16–32 |
| **Cap resolution** | YOLO at 640–960 long side | 720p in |
| **Decode/encode via ffmpeg** | fast I/O, hw accel if present | yes |
| **Async job + progress** | UX hides latency; user isn't blocked | required |
| **Annotate only sampled frames, interpolate** | fewer draws | optional |

Target: 2-min clip processed in **≤90s on one GPU** at N=3, batch=16, 720p.
Output video re-timed so skipped frames are filled (hold last annotation) → looks
smooth at original fps.

## Architecture (request → annotated video)
1. **Browser** uploads video → `POST /jobs/video` (multipart). API stores file to
   a temp/object store, creates a **Job** (id, status=queued), returns `job_id`
   immediately (non-blocking).
2. **JobStore + worker**: async worker (thread now; Celery/RQ later) picks the job,
   runs the **Pipeline**, emits progress (% frames done) into the Job.
3. **Pipeline** (existing, extended):
   decode → detect (YOLO/Roboflow) → track (supervision ByteTrack) → pose
   (MediaPipe, optional) → **fuse** GameState → **annotate** frames (supervision
   annotators) → encode annotated `.mp4` (ffmpeg) → **LLM feedback** (rule/openai)
   → write `result.json`.
4. **Client polls** `GET /jobs/{id}` for status+progress; on `done`, fetches
   `GET /jobs/{id}/video` (annotated mp4) and `GET /jobs/{id}/result` (insights).

## Components / responsibilities
- `src/services/jobs.py` — `Job`, `JobStore` (thread-safe, in-proc; swappable for
  Redis). Status: queued→running→done|error, with `progress` + `result`.
- `src/api/blueprints/jobs.py` — upload + status + result + video-download routes;
  spawns the worker via a `submit()` that takes any processor callable (so the
  lifecycle is testable with a stub — no ML deps here).
- `src/pipeline.py` — add `annotate=True/out_path` + `progress_cb`; supervision
  annotators (lazy). Reuses existing detect/track/fuse/feedback.
- `Config` latency knobs: `FRAME_SKIP` (exists), `BATCH_SIZE` (exists),
  `MAX_LONG_SIDE`, `OUTPUT_FPS`.

## What I build now (💻 verifiable here, no ML deps)
- JobStore + async submit/lifecycle (queued→running→done/error, progress).
- Job API blueprint (upload→job_id, poll, result, video download) — tested with a
  **stub processor** so the whole control plane is green offline.
- Pipeline gains `annotate`/`progress_cb` params + an annotated-video method that
  lazy-imports supervision/cv2 (structurally verified; runs on GPU box).
- Wire Job worker → `Pipeline.process_video`.

## Deferred to 🖥️ GPU box (needs extras)
- Real detect/track/annotate/encode run (the heavy 2-min path).
- GPU batching + hw-accelerated ffmpeg encode.
- Swap thread worker → Celery/RQ + object store (S3) for scale/concurrency.

## Scaling notes (beyond MVP)
- Stateless API + external job queue (Redis) + GPU worker pool (autoscale).
- Object storage for in/out videos; signed URLs to client.
- Per-job GPU memory cap (`Config.GPU_MEMORY_LIMIT` exists); queue backpressure.
- Idempotency: hash input video → cache annotated result (reuse imagehash dedup).

## Acceptance (MVP)
- 💻: upload→job_id→poll→done→download works end-to-end with stub processor; unit
  tests green.
- 🖥️: real 2-min 720p clip → annotated mp4 + insights in ≤ ~90s on one GPU.
