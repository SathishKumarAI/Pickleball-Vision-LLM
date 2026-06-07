# 🧭 Decisions & Thinking Log

> Narrative of the key thinking/decisions on this project + the code patterns that
> resulted. Formal decisions live in `docs/adr/`; this is the "why we got here".

## The arc
1. **Found** a broken-scaffold local copy that didn't match the real GitHub repo →
   synced to `origin/main`, then repaired/dedup'd/import-fixed the Python core.
2. **Made it runnable + safe** (packaging, P0 hardening: GPU gate, input validation,
   timeout/cancel, LLM fallback).
3. **Scoped up** to a production SaaS for ~200 customers — and **right-sized down**
   from the heavy AWS plan to a **managed stack** (the key mind-change).
4. **Built** the managed stack end-to-end (FastAPI + Modal + Supabase + Next.js +
   Stripe + Bedrock), offline-verified (42 pytest + Next build).
5. **Operationalized** — logging/observability, admin log viewer, demo users, infra
   tiers, getting-started, task tracker, and this prompt/skill kit.

## Decisions that shaped the code (with refs)
| Decision | Why | Code |
|---|---|---|
| Managed stack > self-managed AWS | 200 customers is small; minimize ops | [ADR-0005] · `app/`,`worker/`,`web/` |
| FastAPI > Flask | async, Pydantic, OpenAPI, realtime fit | [ADR-0006] · `app/main.py` |
| `rule` default LLM + fallback | can't blow the latency budget; $0 | [ADR-0002] · `src/llm/generate_feedback.py` |
| Cloud LLM = Bedrock, not OpenAI | cost/latency/privacy; user call | [ADR-0003] |
| Reuse OSS (supervision/Roboflow/TrackNet) | don't reinvent | [ADR-0004] · `src/vision/tracking/tracker.py` |
| Browser → Supabase Storage direct | API stays stateless/cheap | [ADR-0007] · `web/components/UploadDropzone.tsx` |
| Lazy heavy imports everywhere | offline-verifiable ~80% of product | `app/routers/analyze.py`, `worker/modal_app.py` |
| Injectable services (Fake/Real) | testable control plane | `app/services/{repo,dispatch,storage}.py`, `app/deps.py` |
| Pure orchestration seam | verify worker logic without GPU | `worker/runner.py` + `tests/test_worker_runner.py` |
| Idempotency (claim_running + sha256) | Modal retries / double-submit safe | `app/services/repo.py`, `app/routers/jobs.py` |

## Reusable code patterns (copy these)
- **Testable router:** `TestClient` + `app.dependency_overrides[get_repo]=…` +
  a dev JWT (`jwt.encode({...,"aud":"authenticated"}, "dev-insecure-change-me")`).
  See `tests/test_jobs_phase2.py`.
- **Service interface + fakes:** `app/services/repo.py` (`Repo` ABC, `InMemoryRepo`,
  `SupabaseRepo`).
- **Worker seam:** inject `analyze_fn` so the GPU model is faked in tests
  (`worker/runner.run_job`).
- **Structured logging:** `app/logging_config.py` (JSON + CLI + ring buffer +
  correlation id).
- **Court analytics (pure numpy):** `src/integration/analytics/`.

## What's deliberately deferred (and why)
- **M0 GPU seam run** — needs Modal+GPU; everything downstream assumes it works.
- **Vision-model upgrades** (BoT-SORT/TrackNet/action/NVENC) — config-flagged,
  GPU-side, ship after the seam.
- **CI** — wire the offline pytest + Next build on push.

## How to keep going
Use `.claude/skills/ship-feature` + the prompts in `.claude/docs/PROMPT_LIBRARY.md`.
Track in `docs/TASKS.md`.
