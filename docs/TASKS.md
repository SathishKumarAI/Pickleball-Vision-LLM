# ✅ Project Task Tracker

> Living checklist of all tasks + sub-tasks with status, kept updated as work
> proceeds. Status: ✅ done · 🔄 in progress · ⬜ todo · 🖥️ needs GPU/cloud.
> Last updated 2026-06-07.

## Legend
✅ done & verified (offline pytest / Next.js build) · 🖥️ needs Modal/Supabase/GPU
to run · ⬜ not started. Commit hashes reference `main`.

---

## EPIC 1 — Repo repair & baseline (pre-migration) ✅
- [x] Flatten broken scaffold, sync to real GitHub `main`
- [x] Package (`pyproject`, `src` layout), runnable app
- [x] De-duplicate `src/`; vision import-coherence (17 broken imports → 0)
- [x] Pipeline glue: fusion → feedback → `analyze_detections`/`process_video`
- [x] OSS reuse (supervision ByteTrack), MODELS_AND_REUSE doc
- [x] P0 hardening (GPU gate, input validation, timeout/cancel, LLM fallback)
- [x] Cloud LLM = Bedrock (OpenAI removed/commented), budget plan

## EPIC 2 — Managed-stack migration (production build)
### Phase 0 — FastAPI foundations ✅ `e31ad8f`
- [x] `app/` package + Pydantic `Settings`
- [x] Supabase JWT verify (HS256 + JWKS) + `get_current_user`/`require_admin`
- [x] Routers: health, auth/me, analyze (reuses `analyze_detections`)
- [x] Supabase schema migration (profiles/jobs/analyses/subscriptions/usage + RLS + trigger)
- [x] 8 offline tests

### Phase 1 — Modal GPU worker ✅ `c882eed`
- [x] `worker/runner.run_job` (idempotency, download→analyze→upload, progress, cancel, error, usage)
- [x] `worker/modal_app.py` (GPU function, weights Volume, build vision Config only here)
- [x] repo `claim_running` + storage file transfer
- [x] 5 worker tests (success/idempotency/cancel/error)
- [ ] 🖥️ **M0 seam run** — one real clip end-to-end on Modal GPU + record timings

### Phase 2 — Jobs control plane ✅ `fb95996`
- [x] repo (InMemory + Supabase), dispatch (Fake + Modal), storage (Fake + Supabase), quota
- [x] jobs router: upload-url, create (quota→dedup→spawn), list, get, result, video, cancel, quota
- [x] ownership enforcement + 10 tests

### Phase 3 — Next.js frontend ✅ `0d2f9e4`
- [x] Auth (/login), guarded `(app)` group, Nav
- [x] Pages: landing, dashboard, upload, jobs/[id] (live realtime), analyses/[id], history, billing, settings, admin
- [x] lib: supabase clients, api wrapper, useJobRealtime; direct-to-storage upload
- [x] `npm run build` clean (10 routes)

### Phase 4 — Stripe billing + quota ✅ `5b5052d`
- [x] checkout/portal/webhook; pure `handle_event`; quota gate on /jobs (402)
- [x] tests (upgrade/downgrade/ignore)

### Phase 5 — Account / admin / GDPR / observability / retention ✅ `5b5052d`,`8cfdeae`,`12b1fb7`
- [x] /account usage + DELETE (GDPR erase); /admin/jobs (admin-gated)
- [x] structured logging (JSON + Ansible/CLI console + ring buffer) + correlation id
- [x] /admin/logs + admin UI live log viewer
- [x] retention cron (Modal) + optional Sentry
- [x] demo users seed script + DEMO_ACCESS doc

### Phase 6 — Vision correctness ✅ (math) / 🖥️ (models) `8e8450d`
- [x] court homography (pure NumPy DLT) + analytics (heatmap/kitchen/rally tempo) wired into results
- [ ] 🖥️ BoT-SORT ReID tracker (config flag)
- [ ] 🖥️ TrackNet ball backend
- [ ] 🖥️ action/shot classifier (replaces `_infer_action`)
- [ ] 🖥️ NVENC hardware encode

## EPIC 3 — Docs ✅ `8bccf52`,`d2ccc12`
- [x] RFC-003 managed stack (supersedes RFC-002), ADRs 0001–0007
- [x] ROADMAP/BUDGET/DELIVERY superseded banners (content preserved)
- [x] INFRA_SCALING (min→max tiers), DEMO_ACCESS, TASKS (this file)
- [🔄] GETTING_STARTED (requirements to use/test) — in progress
- [🔄] Prompt library + Claude skill + memory (Anthropic-style) — in progress

---

## Remaining to go live (the short list)
1. 🖥️ Provision Supabase + Modal + Vercel; apply migrations; deploy `app/`+`worker/`+`web/`.
2. 🖥️ **M0 seam run** (real clip → annotated video) + record per-stage timings/cost.
3. 🖥️ Seed demo users (`scripts/seed_demo_users.py`).
4. 🖥️ Vision-model upgrades (Phase 6 GPU flags) once the seam is proven.
5. ⬜ Wire CI to run the offline pytest suite + Next.js build on push.

## How this is tracked
Live during a session via the Task tool; this file is the durable mirror. Update
the boxes + commit hashes as phases complete.
