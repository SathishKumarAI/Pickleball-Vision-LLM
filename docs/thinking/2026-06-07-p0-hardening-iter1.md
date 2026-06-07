# Thinking — P0 hardening, iteration 1 (2026-06-07)

From the architecture review. Implemented the P0 items verifiable offline; Redis
(P0-3) + Celery/S3 (P1) deferred — need infra. See docs/REMEDIATION_PLAN.md.

## Done (✅ verified, no ML deps)
- **P0-1 GPU gate** — `src/services/gpu_gate.py`: `BoundedSemaphore(GPU_SLOTS=1)` +
  `run_on_gpu(fn, timeout_s)`. `process_video` now runs `detector.detect` through
  it → concurrent jobs queue instead of OOMing one GPU. `GpuBusyError` on admission
  timeout. Verified.
- **P0-2 Input validation** — `src/api/validate.py`: ffprobe size/duration/codec/
  resolution checks wired into `POST /jobs/video` (413/415/422). **Tested against the
  real 76MB sample** → correctly rejected (297s > 180s budget); too-big→413; empty→422.
- **P0-4 Timeout + cancel** — `JobStore.request_cancel`/`is_cancelling`, `JobCancelled`,
  statuses CANCELLING/CANCELLED, `POST /jobs/<id>/cancel`. `process_video` polls
  `cancel_cb` + enforces `max_seconds` wall-clock (default 450s = 5× budget). Fixed a
  race: queued-then-cancelled no longer clobbered by RUNNING. Both pre-submit and
  during-run cancel verified → `cancelled`.
- **P0-5 LLM budget fallback** — `CoachingFeedbackGenerator(deadline_s, fallback)`:
  openai/hf calls run under a wall-clock; on timeout/error → rule coach (never blows
  the budget or fails the job). Verified: openai backend, no key, 0.5s deadline →
  returned rule text, no raise.

## Deferred (need infra/GPU — documented in REMEDIATION_PLAN.md)
- P0-3 Redis JobStore (no redis here) · P1-1 S3 · P1-2 Celery · P1-4 metrics ·
  P1-5 lifecycle/DELETE · P1-6 rate-limit/token-expiry · P2-* vision correctness.

## Open decision — LLM backend (user asked "why openai")
Default backend is already `rule` (zero deps, no external calls). `openai` is one
*optional* hosted backend; `hf` runs local transformers. Given the OSS-first
directive + the P0-5 cost/latency concern, recommend: make **local HF (Mistral/
LLaMA)** the canonical real backend, keep OpenAI opt-in only. No code rip needed —
just default/emphasis. Next iteration if confirmed.
