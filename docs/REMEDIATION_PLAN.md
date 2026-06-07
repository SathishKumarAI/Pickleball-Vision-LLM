# Remediation Plan — production-hardening the video pipeline

> From an architecture review of `docs/assets/system_design.png`. Sequenced so
> each step unlocks the next. Status tags: ✅ done (verified offline) · 🖥️ needs
> GPU box · 🧰 needs infra (Redis/S3/Celery).

## Two root causes (fix these and most gaps resolve)
1. **Work runs in-process** — a Flask background thread runs `process_video`. Cause
   of: no concurrency control, no durability, no timeout, no backpressure, jobs lost on restart.
2. **State + files are box-local** — `JobStore` in RAM, outputs in a local workdir via
   `send_file`. Cause of: scale-out breaks step 6, no persistence.

Migration order: **bound the GPU → externalize state → externalize work → externalize files.**

---

## P0 — before any real users

- **P0-1 GPU concurrency gate** ✅ — `src/services/gpu_gate.py`: `BoundedSemaphore(GPU_SLOTS=1)` + `run_on_gpu(fn, timeout_s)`. Wrap detect/track/pose/annotate. Stops N-uploads → 1-GPU OOM. (Semaphore now; queue depth in P1-2.)
- **P0-2 Input validation at upload** ✅ — `src/api/validate.py`: `ffprobe`-based size/duration/codec/resolution checks before accepting a job. Rejects 4-hour/4K DoS + malformed media. Wired into `/jobs/video` (413/415/422).
- **P0-3 Durable job state** 🧰 — swap in-RAM `JobStore` → Redis hash-per-job + TTL. Survives restart, single source of truth. (Needs redis.)
- **P0-4 Timeouts + max-runtime kill + cancel** ✅ — subprocess `timeout=`; whole-job wall-clock ceiling; cooperative cancel flag (`POST /jobs/<id>/cancel` → worker polls). Hard-kill of wedged GPU op = P1-2 (Celery revoke).
- **P0-5 LLM step can't blow the budget** ✅ — `generate_feedback`: rule-based path is a real **timeout/error/cost fallback**, not an alternative. Confirm BLIP-2+YOLO+ByteTrack+MediaPipe co-fit GPU mem (🖥️); cold model-load not in the 90s.

## P1 — reliability & scale-out

- **P1-1 Outputs → object storage + signed URLs** 🧰 — worker writes `annotated.mp4`/`result.json` to S3; step 6 returns a short-lived presigned URL. Load-bearing for >1 worker.
- **P1-2 In-process thread → task queue** 🧰 — Celery/RQ, dedicated `gpu` queue, `--concurrency=<gpu_slots>`, `soft_time_limit`/`time_limit`, `acks_late=True` (crash → requeue). Autoscale on queue depth; **pre-warm models**.
- **P1-3 Retry / idempotency / error taxonomy** 🧰 — classify retryable (OOM, API 5xx) vs permanent (bad input); per-frame try/except (drop+hold-last); content-hash idempotency → cache result, skip GPU.
- **P1-4 Observability** 🖥️ — p95 end-to-end + per-stage latency (decode/detect/track/annotate/encode/coach), GPU mem/util, queue depth, jobs-by-status counter, alerts near SLA. Prometheus/Grafana (configs already in `deployment/`).
- **P1-5 Data lifecycle / GDPR** 🧰 — TTL on Redis keys + S3 lifecycle expiry; clean intermediate frames in `finally`; `DELETE /jobs/<id>`; stated retention policy (videos contain faces).
- **P1-6 Auth hardening** ⏳ — rate-limit/lockout on `/login` `/register`; token expiry + revocation (`jti` set in Redis); confirm ownership enforced on every job route.

## P2 — vision correctness (pickleball) & UX

- **P2-1 Court homography** 🖥️ — image→top-down court via known court dims; positions/shot-placement in real units. Biggest coaching-correctness gap.
- **P2-2 Don't hold-last the ball** 🖥️ — ball is small/fast; run ball every frame or interpolate; players keep hold-last.
- **P2-3 Re-ID across track switches** 🖥️ — appearance embedding or court-side priors to re-anchor IDs after occlusion.
- **P2-4 Multi-person pose** 🖥️ — crop per-track bbox → single-person pose, or multi-person model.
- **P2-5 Name the action model** 🖥️ — temporal classifier over pose+ball window; until trained, label actions as stub.
- **P2-6 Job listing + notifications** ⏳ — `GET /jobs` owner-scoped paginated; webhook/email on completion.

---

## Execution order
1. P0-1 gate → stops OOM. ✅
2. P0-2 validation → stops unbounded-job DoS. ✅
3. P0-3 Redis JobStore → durability + foundation. 🧰
4. P0-4 / P0-5 timeouts + LLM fallback → SLA stops being a lie. ✅
5. P1-1 / P1-2 S3 + Celery → scale-out works; thread root-cause gone. 🧰
6. P1-3..6 retries, metrics, lifecycle, auth → production-grade.
7. P2-* vision correctness → coaching gets good. 🖥️

> **Prereq for trusting any of this:** wire the stub control plane to the real GPU
> pipeline end-to-end. The seam (file handoff, progress callbacks under real timing,
> memory pressure, true encode cost) has never run together.

## Done this iteration (✅, verified offline, no ML deps)
P0-1 GPU gate · P0-2 ffprobe input validation (tested vs real sample) · P0-4 cancel
+ wall-clock timeout · P0-5 LLM timeout/cost fallback.

---

## References / Further reading
- Task queue: [Celery docs](https://docs.celeryq.dev/) · [Celery `acks_late`/time limits](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- Concurrency: [Python `threading.BoundedSemaphore`](https://docs.python.org/3/library/threading.html#threading.BoundedSemaphore)
- Object storage: [S3 presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html) · [S3 lifecycle (TTL)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)
- State/queue: [ElastiCache for Redis](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html) · [Amazon SQS](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html)
- Media validation: [ffprobe docs](https://ffmpeg.org/ffprobe.html)
- Observability: [Prometheus](https://prometheus.io/docs/introduction/overview/) · [CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html)
- Tracking/ReID (P2-3): [Datature — ByteTrack](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box) · [soccer ByteTrack+ReID](https://github.com/Anudeep007-hub/soccer-multi-object-tracking)
- Court homography (P2-1): [Roboflow — camera calibration with keypoints](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)
- Privacy: [GDPR overview](https://gdpr.eu/what-is-gdpr/)
- Internal: `docs/DELIVERY_PLAN.md`, `docs/specs/RFC-001-video-analysis-pipeline.md`, `docs/specs/RESEARCH_NOTES.md`
