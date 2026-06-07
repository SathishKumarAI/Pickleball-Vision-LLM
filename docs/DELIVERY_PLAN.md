# Delivery Plan — AWS · ~10k videos/mo · rule-first MVP

> Concrete execution plan built on locked decisions (2026-06-07). Complements the
> milestone view in `docs/ROADMAP.md`. No code here — this is the build order.

## Locked decisions
- **Cloud:** AWS. **Scale:** ~10k videos/mo. **MVP coaching:** `rule`-only (cloud
  LLM = AWS Bedrock Claude Haiku, added post-MVP).
- Default LLM stays `rule` (free, no provider dependency for v1); `hf` local for dev.

---

## Target AWS architecture (concretized)

| Concern | AWS choice | Notes |
|---|---|---|
| Web/API | ECS Fargate (Flask app, 2+ tasks) behind **ALB** | stateless; scales on CPU |
| Auth/users | **RDS Postgres** (small) | replaces SQLite `UserDB`; same interface |
| Job state | **ElastiCache Redis** | replaces in-proc `JobStore`; TTL keys |
| Queue | **Celery on Redis** (or SQS) | dedicated `gpu` queue, depth-limited |
| GPU workers | **ECS/EC2 g5.xlarge (A10G)**, autoscale on queue depth, **Spot** | pre-warm models; concurrency = real GPU slots |
| Object store | **S3** (`uploads/`, `jobs/<id>/annotated.mp4`, `result.json`) | presigned URLs; lifecycle TTL |
| CDN/egress | CloudFront on the S3 output bucket | cheap annotated-video delivery |
| Observability | CloudWatch metrics+logs (+ Prometheus/Grafana optional) | p95 latency, GPU util, queue depth |
| Secrets | SSM Parameter Store / Secrets Manager | `APP_SECRET`, Bedrock creds |
| Model weights | S3 + cached on worker volume | Roboflow pickleball `.pt`; versioned keys |

At 10k/mo: ~1–2 g5 Spot workers steady, autoscaling on bursts. Budget ≈ **$270/mo**
(`docs/BUDGET_PLAN.md`); rule-only MVP drops the LLM line → ≈ **$120/mo**.

---

## M0 — Prove the seam (🖥️ GPU box) — THE blocker

Goal: run the real `Pipeline.process_video` through the job control plane on one
GPU box, end to end, with the sample clip. Until this runs, every estimate is theory.

**Steps**
1. Provision a g5.xlarge (or any A10G/T4 box). `pip install -e ".[vision]"`.
2. Set `Config.DETECTOR_MODEL` to a Roboflow pickleball weight (GameChangerv1).
3. Run `Pipeline(tracker_backend="supervision").process_video(sample, annotate_path=...)`
   directly — confirm detect→track→annotate→encode produces a valid mp4.
4. Run it **through** `/jobs/video` (local Flask) → poll → download. Exercise the
   real seam: file handoff, `progress_cb` timing, cancel, GPU gate.
5. **Measure** per-stage wall-clock (decode/detect/track/annotate/encode) + total,
   peak GPU memory with all models loaded, cold model-load time.

**Exit criteria**
- Sample clip → annotated mp4 + `result.json` via the API.
- Per-stage timings recorded; total within (or extrapolating to) the ≤90s budget at
  FRAME_SKIP=3, 720p.
- Confirmed YOLO+ByteTrack(+pose) co-fit in g5 memory; cold-load time documented.
- Cancel + timeout verified on a real run.
- **Output:** a `docs/thinking/M0-seam-results.md` with the numbers.

**Risks this surfaces:** real encode cost, memory co-fit, progress-% accuracy across
stages, ffmpeg codec edge cases on real uploads.

---

## Sprint breakdown (2-week sprints)

### Sprint 1 — Seam + test foundation
- M0 seam test on a GPU box (above). *(highest priority)*
- **G1:** pytest suite for all offline-verified behavior (game_state, feedback+fallback,
  tracker, jobs lifecycle+cancel, auth+ownership, validate, gpu_gate, api); fix the
  stale `.github/workflows/ci-cd.yml` (py3.11/3.12, `pip install .[dev]`, pytest+ruff).
- **Exit:** green CI; one real clip processed via the API; M0 results doc.

### Sprint 2 — Durable single-box → AWS skeleton
- Redis `JobStore` (P0-3) behind the existing interface; ElastiCache.
- RDS Postgres for `UserDB` (same interface; SQLAlchemy or psycopg).
- S3 output + presigned URLs (P1-1); replace `send_file`.
- Terraform/CDK for ALB+ECS(api)+Redis+RDS+S3 (no GPU autoscale yet).
- **Exit:** app runs on AWS (API on Fargate), jobs durable, outputs in S3.

### Sprint 3 — Task queue + GPU workers
- Celery on Redis, dedicated `gpu` queue; replace in-proc thread (P1-2).
- ECS/EC2 g5 Spot worker pool, autoscale on queue depth; model pre-warm.
- Retry taxonomy + content-hash idempotency (P1-3); per-frame isolation.
- **Exit:** concurrent uploads processed safely under load; crash → requeue;
  scale-out demonstrated (2 workers).

### Sprint 4 — Observability + lifecycle + auth hardening
- CloudWatch (or Prom/Grafana): p95 + per-stage latency, GPU util, queue depth,
  jobs-by-status; alarms near SLA (P1-4).
- Data TTL/GC, S3 lifecycle, `DELETE /jobs/<id>`, retention policy (P1-5).
- Rate-limit `/auth/*`, token expiry + revocation list in Redis (P1-6).
- **Exit:** SLA dashboard live; data lifecycle enforced; auth hardened.

### Sprint 5 — Web UX (rule-MVP launch-ready)
- Dashboard (Streamlit first): login → upload → progress → annotated video +
  states + feedback (P2-6/F2); poll now, SSE later.
- `GET /jobs` history (done), completion webhook/email.
- **Exit:** end-to-end product demo; rule-coaching MVP shippable.

### Sprint 6+ — Vision correctness, then cloud LLM
- M4: court homography (biggest coaching gap) → action classifier → re-ID →
  multi-person pose → ball-every-frame.
- Add Bedrock Claude Haiku `cloud` backend behind the existing fallback; A/B vs `rule`.
- **Exit:** coaching is court-aware and model-driven; cloud LLM opt-in live.

---

## Critical path
M0 seam → Sprint 2 (durable+S3) → Sprint 3 (queue+GPU scale) are the load-bearing
sequence; everything else parallels or follows. **Do M0 before committing to any
sprint dates** — its numbers set the real GPU pool size and SLA feasibility.

## Definition of done (per sprint)
Tests green in CI · deployed to a staging AWS env · a `docs/thinking/` iteration
note with decisions + measured numbers · roadmap milestone checkboxes updated.

## Still-open (smaller) questions
- Data retention window (days) for uploaded videos → sets S3 TTL (GDPR).
- IaC tool: Terraform vs AWS CDK.
- Auth: keep custom bearer tokens vs adopt Cognito (custom is fine for MVP).
