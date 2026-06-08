# 📌 Project Status — Done / To-Do (handoff)

> Snapshot to pick up later. Last updated 2026-06-08. Everything below is on
> `main` and offline-verified (52 pytest + Next.js build green) unless marked 🖥️
> (needs GPU/cloud) or 🧰 (needs managed infra).

## TL;DR
A production-grade pickleball video-analysis **SaaS is built end-to-end** on a
managed stack and verified offline. The **one blocker to going live** is running the
real GPU pipeline once (the "M0 seam"). The vision-intelligence (analytics) layer is
built on OSS and tested; real model inference is GPU-only.

---

## ✅ Done (built + verified offline)

### Backend / control plane (`app/`) — FastAPI
- Auth (Supabase JWT verify), jobs (upload-url → quota → dedup → Modal spawn →
  list/get/cancel/result/video), billing (Stripe checkout/portal/webhook), account
  (usage + GDPR delete), admin (all-jobs + live log viewer).
- Injectable services (repo / dispatch / storage) with in-memory + Fake impls →
  the whole control plane tests with no cloud/GPU.
- Structured logging (JSON + ansible/CLI console + ring buffer + correlation id).

### Worker (`worker/`) — Modal
- `runner.run_job` (idempotency, download→analyze→upload, progress, cancel, error,
  usage) — pure, tested. `modal_app.py` wraps `Pipeline.process_video` + retention cron.

### Data (`supabase/migrations/`)
- profiles/jobs/analyses/subscriptions/usage + RLS + new-user trigger; storage RLS.

### Frontend (`web/`) — Next.js + Tailwind
- "Athletic Editorial" brand landing (Anton/Fraunces/Spline Sans) + product-register
  app (panels, status pills, skeletons, empty states); admin log viewer; impeccable
  craft pass (AA contrast, reduced-motion, focus). `npm run build` clean.

### Vision-intelligence (`src/vision/analysis/`) — OSS, no GPU
- `trajectory.py` (scipy savgol, bounce detection), `actions.py` (shot classifier,
  PB-Vision taxonomy, sklearn-pluggable), `rally.py` (rally/shot segmentation).
  Wired into `pipeline` → `result["intelligence"]`.

### Vision models (`src/vision/`) — lazy adapters (parse-verified, 🖥️ to run)
- Roboflow detector, supervision ByteTrack + **BoT-SORT/boxmot ReID**, TrackNet ball.

### Docs
- ROADMAP · RFC-001/002/003 · ADRs 0001–0007 · INFRA_SCALING · BUDGET · GETTING_STARTED
  · DEMO_ACCESS · TASKS · MODELS_AND_REUSE · `docs/inspiration/pb-vision/*` (competitor
  metric set + build mapping) · `.claude/skills/ship-feature` + prompt library.

---

## ⬜ To-Do (next, by priority)

### P0 — make it real (🖥️/🧰, blocked on accounts)
1. **M0 GPU seam** — provision Supabase + Modal + Vercel; run ONE real clip through
   `/jobs/video` → Modal → Storage → UI; record per-stage timings/GPU mem/cold-start
   (`docs/thinking/M0-seam-results.md`). Everything's latency/cost assumptions ride on this.
2. Seed demo users (`scripts/seed_demo_users.py`); deploy `app/`+`worker/`+`web/`.

### P1 — vision intelligence build-out (mostly offline; per `docs/inspiration/pb-vision/build-mapping.mdx`)
3. **Trajectory outlier rejection** (speed-gate / RANSAC) — real-footage test showed a
   naive detector yields 353 px/f junk; make analysis noise-robust. *Offline-testable.*
4. **Shot speed (mph)** via court-homography scale; **net/out error** detection.
5. **Skill-rating composite** (sklearn; fixed-weight first) — `src/vision/analysis/skill.py`.
6. **Highlights** (rally length / events → ffmpeg cut) 🖥️; kitchen-reach + court-coverage
   (needs player tracking) 🖥️.

### P2 — platform
7. **CI**: run offline pytest + `web` build on push (fix the stale `.github/workflows`).
8. Async job API polish (SSE progress), `GET /jobs` history (done) + webhooks.
9. Observability dashboards (CloudWatch/Sentry) once deployed.

---

## ⚠️ Gotchas / constraints
- **This machine = Python 3.14, no torch/cv2/GPU.** Only control plane + numpy/scipy/
  sklearn analysis run here. Vision models run on Modal/GPU.
- **Don't `next build` while the dev server runs** on the same `.next` (corrupts chunks).
  Run dev with `cd web && ./node_modules/.bin/next dev -p 3000`.
- Real coaching accuracy is gated on a **trained detector** (Roboflow/YOLO + TrackNet),
  not the heuristic — proven by the real-footage test.

## How to resume
Read this + `memory/project-state`. For a feature use the **ship-feature** skill.
Start P1-3 (outlier rejection) for an offline win, or P0-1 (M0 seam) when cloud/GPU
access is ready. Test matrix in `docs/GETTING_STARTED.md`.
