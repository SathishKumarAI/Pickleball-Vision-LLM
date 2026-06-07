---
name: ship-feature
description: >-
  Ship a new end-to-end feature in the Pickleball-Vision-LLM managed-stack SaaS
  (FastAPI + Modal + Supabase + Next.js). Use when adding any product capability —
  a new API route, a worker step, a vision/analytics module, or a UI page. Enforces
  the project's patterns (lazy heavy-imports, injectable services, offline-first
  verification, docs + ADR, commit+push per task). Trigger: "add a feature",
  "build <X> endpoint/page/worker step", "ship <X>".
---

# Ship a feature (Pickleball-Vision-LLM)

Drive a feature from idea → verified → pushed, following the conventions already
established in this repo. Keep the offline-verifiable surface green at every step.

## Non-negotiable patterns (why they exist)
1. **Lazy heavy imports.** torch/cv2/ultralytics/supabase/modal/stripe import
   *inside* functions, never at module top of `app/`. → the API + tests run with
   no ML/cloud deps; ~80% of the product is verifiable offline.
2. **Injectable services.** New external dependency → add an interface in
   `app/services/` with an `InMemory`/`Fake` impl + a real impl, selected in
   `app/deps.py`. → tests use fakes via `app.dependency_overrides`.
3. **Reuse the core.** `src/` is the portable library (pipeline/fusion/llm/vision).
   Reuse it; don't fork. Prefer OSS over custom (see `docs/MODELS_AND_REUSE.md`).
4. **Verify before claiming done.** Run `pytest` (offline) + `npm run build` (web).
   Heavy GPU/cloud paths are parse-checked + documented for the GPU box.
5. **One commit per task + push.** Conventional commit; end body with the
   Co-Authored-By line. Update `docs/TASKS.md`.

## Workflow
1. **Locate** — find the analogous existing code (router/service/worker/page) to
   mirror. Examples: API route → `app/routers/jobs.py`; service → `app/services/
   repo.py`; worker logic → `worker/runner.py`; UI page → `web/app/(app)/*`.
2. **Design** — if it spans >2 files or adds a dependency, write/append a short
   note in `docs/thinking/` and (for a real decision) an ADR in `docs/adr/`.
3. **Build** — implement following the 5 patterns. Add the route to
   `app/main.py`; add Pydantic models to `app/models`; add fakes for new services.
4. **Test** — add `tests/test_*.py` using `TestClient` + dependency overrides
   (mint a dev JWT with `dev-insecure-change-me`). Cover happy path + auth +
   ownership + edge. Run `pytest -q` in `.venv`.
5. **Frontend (if any)** — page under `web/app/(app)/`, call the API via
   `lib/api.ts`, live data via `useJobRealtime`/Supabase. `npm run build` must pass.
6. **Docs** — update `docs/TASKS.md` status; add a `References` section to any new
   doc; keep superseded content (comment/move, never delete).
7. **Ship** — `git add` the specific files, commit, `git push origin main`.

## Definition of done
`pytest` green · `npm run build` green (if UI) · docs updated · committed + pushed ·
GPU/cloud-only parts clearly marked 🖥️ and parse-checked.

## Reference docs
`docs/GETTING_STARTED.md` · `docs/specs/RFC-003-managed-stack.md` ·
`docs/INFRA_SCALING.md` · `.claude/docs/PROMPT_LIBRARY.md` (prompts to drive this).
