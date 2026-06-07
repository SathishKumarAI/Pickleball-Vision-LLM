# 🏓 Pickleball-Vision-LLM — Engineering Plan

> Generated 2026-06-06 from the live repo at commit `d3e39ef` (origin/main).
> Supersedes any earlier `Docs/PLAN.md` (that was written against a stale local
> snapshot and is obsolete). This plan reflects the **actual** codebase.

---

## 📍 Current State

Repo is **mid-reorganization** (latest commit: _"needs to clean and more changes"_;
`scripts/` holds ~15 one-off `reorganize_*/cleanup_*` scripts; `backup/` holds 6
tracked snapshots). Real vision code exists, but structure is duplicated, the
entry point is broken, and there is no installable package.

| Area | State |
|------|-------|
| Vision (`src/vision/detection/`) | 🟢 Real code — `detector.py` (234L), `collect_data.py`, `preprocessor.py`, `data_collector.py`, `metrics.py`, `visualization.py` |
| Entry point `src/main.py` | 🔴 **Broken** — `from src.api import create_app`; `src/api/` does not exist (real API lives in `src/temp/api/`) |
| `core/utils` vs `fusion/utils` | 🔴 Duplicated — `alerts.py`, `video_utils.py`, `logger.py`, `rename.py` byte-identical; `metrics.py` near-dup |
| `src/llm/train_model.py` vs `training/train_model.py` | 🔴 Identical duplicate |
| `src/vision/detection/tracker.py` vs `tracking/tracker.py` | 🔴 Identical duplicate |
| `src/vision/{detection,preprocessing,tracking,postprocessing}` | 🟡 Overlapping — `preprocessor.py` in both `detection/` and `preprocessing/` |
| `src/llm/src/` | 🟡 Odd nested pkg (`clip_integration.py`, `video_llava_connector.py`, `prompt_templates.py`) |
| Stub packages | 🟡 Empty `__init__.py` only: `integration/{analytics,fusion,streaming}`, `llm/{analytics,embeddings,models,prompts}`, `vision/postprocessing`, `web` |
| `src/temp/` | 🟡 Temp API code never homed |
| Packaging | 🔴 No root `pyproject.toml` / `setup.py` / `requirements.txt` / `Makefile` → package not installable; absolute `src.*` imports won't resolve without it |
| Ops config | 🟢 In `deployment/` — `Dockerfile`, `docker-compose[.prod].yml`, `Makefile`, `.pre-commit-config.yaml`, monitoring (prometheus/alertmanager) |
| `backup/` | 🔴 **1073 files / 10 MB tracked in git** — not gitignored, bloats clones |
| `docs/reorganization_plan.md` | 🟡 Stale — describes `src/pickleball_vision/`, `src/app/`, `src/frontend/` that no longer exist |
| Root `README.md` | 🔴 Missing (only `src/README.md`, `docs/README.md`) |

---

## 🎯 Phase 0 — Repo Hygiene (do first, low risk)

- [ ] **0.1** Stop tracking `backup/` — add to `.gitignore`, `git rm -r --cached backup/` (keep on disk). Removes 1073 files / 10 MB from the tree.
- [ ] **0.2** Add root `README.md` (project overview + quickstart). Pull vision from `docs/PROJECT_SUMMARY.md`.
- [ ] **0.3** Decide fate of `scripts/` one-off reorg scripts (`reorganize_*`, `organize_*`, `cleanup_*`, `smart_organizer.py`) — archive to `scripts/_archive/` or delete; they were single-use.
- [ ] **0.4** Refresh or delete `docs/reorganization_plan.md` (currently describes a layout that no longer exists).

## 🧱 Phase 1 — Make It Runnable (BLOCKER for everything downstream)

- [x] **1.1** Add root `pyproject.toml` — light core deps (flask, dotenv, pydantic, pyyaml, numpy, tqdm) + optional extras `[vision]`/`[llm]`/`[mlops]`/`[dev]`; `src` layout; `pickleball` console script.
- [x] **1.2** Fix entry point. Decision: **Flask** — `src/main.py` already used the Flask factory+`app.run(debug=)` idiom; `temp/api` (FastAPI) is a broken relic of the old `src/pickleball_vision` layout. Created `src/api/__init__.py` with a `create_app()` factory (`/health`, `/`). Fixed `setup_logger()` → `setup_logger("pickleball_vision")` (name is required). Made `settings.py` tolerate missing `python-dotenv`.
- [x] **1.3** Verified `settings.py` exports `API_HOST`/`API_PORT`/`DEBUG`.
- [x] **1.4** Smoke test passes: factory builds, `/health`→200, `src.main` imports resolve.
- [x] **1.5** Added `requirements.txt` (light core, mirrors pyproject) + `docs/SETUP.md` (conda/venv quickstart, extras matrix, `uv` path, env-var config, test + deploy pointers).
- [ ] **1.6** (follow-up) Wire the real detection endpoint from `temp/api/app.py` into a Flask blueprint once `detector`/`preprocessor` modules are consolidated (Phase 2/3). FastAPI relic in `temp/` to be removed.

## 🧹 Phase 2 — De-duplicate (reduce maintenance surface)

- [x] **2.1** Merged `src/fusion/utils/` → canonical `src/core/utils/`. Identical copies deleted; unique `data_collector.py` + `visualization.py` moved to core; the two **non-identical** `metrics.py` reconciled — core had classification/pose/game-state + a stubbed tracking; fusion had the real IoU/detection/tracking — both function sets now live in `core/utils/metrics.py`. `fusion/utils/` removed. (No external importers existed.)
- [x] **2.2** Removed duplicate `src/llm/train_model.py` (kept `src/llm/training/train_model.py`). Byte-identical, zero importers.
- [x] **2.3** Removed duplicate `src/vision/detection/tracker.py` (kept `src/vision/tracking/tracker.py`). Byte-identical, zero importers.
- [x] **2.5** Flattened bogus `src/llm/src/` nest — `clip_integration.py`, `generate_feedback.py`, `prompt_templates.py`, `video_llava_connector.py` moved up to `src/llm/`.
- [x] **2.4** Resolved (in Phase 3). Canonical `FramePreprocessor` = `vision/detection/preprocessor.py` (Config-compatible, more complete); incompatible `vision/preprocessing/preprocessor.py` removed; `detector.py` now imports the sibling `.preprocessor` and the canonical Config.
- [x] **2.6** Done. `src/temp/` (broken FastAPI relic, 7 dangling imports, zero importers) removed. Endpoint pattern preserved in git history; rebuild as a Flask blueprint in Phase 3.2+ if needed (supersedes task 1.6).

> Per the "use OSS, don't reinvent" directive: left a TODO in `metrics.py` to
> swap hand-rolled `calculate_iou` for `torchvision.ops.box_iou` / `supervision`
> once those deps are installed and verifiable.

## 🔌 Phase 3 — Wire the Pipeline (fill the stubs)

- [x] **3.1** Vision path is now **import-coherent**. Fixed the whole broken-import web: canonical Config moved `fusion/config/config.py` → `src/core/config/config.py` (stray `..logging.logger` import fixed); `detector.py` repointed (`shared.config`→`core.config.config`, `utils.preprocessor`→sibling `.preprocessor`); `video_processor.py` + both llm scripts repointed off the dead `pickleball_vision.*` / `vision.core.*` paths. **Static import scan: 17 broken → 0.** All `src/` parses; app still boots (`/health`→200). Legacy `test_collection.py` moved to `tests/` with a skip guard. (Runtime exec of vision modules still needs the `[vision]` extras — torch/cv2/ultralytics.)
- [ ] **3.2** Implement empty `llm/{models,prompts,embeddings,analytics}` stubs — game-state → prompt → coaching feedback. (Needs `[llm]` extras to verify.)
- [ ] **3.3** Implement `integration/{fusion,streaming,analytics}` — connect vision output to LLM input (`fusion` module is the join point).
- [ ] **3.4** End-to-end orchestrator: one entrypoint runs a clip through vision + LLM → feedback. Re-add a `/detect` Flask blueprint (pattern preserved from the removed `temp/api`).

## 🧪 Phase 4 — Tests, CI, Quality

- [ ] **4.1** Get `scripts/tests/` (and any `tests/`) running under `pytest`; fix import paths post-refactor.
- [ ] **4.2** Wire `.github/workflows/` to run lint + tests on push (verify the 2 existing workflows actually work).
- [ ] **4.3** Activate `deployment/.pre-commit-config.yaml` at root (or move it) — format/lint gate.
- [ ] **4.4** Add coverage for the de-duplicated utils + detector.

## 🚀 Phase 5 — Deploy & Docs

- [ ] **5.1** Validate `deployment/Dockerfile` + `docker-compose.yml` build against the fixed package.
- [ ] **5.2** Verify monitoring stack (prometheus/alertmanager configs) wires to a running service.
- [ ] **5.3** Consolidate `docs/` (`technical/` has ~30 files) — single navigable index; drop stale docs.

---

## ⚠️ Risk Notes

- **Imports are fragile.** Many `from src.x.y` absolute imports + no packaging = nothing runs outside repo root. Phase 1.1 unblocks Phases 2–5.
- **Refactor order matters.** Do Phase 1 (runnable + a test harness) before Phase 2 dedup, so moves are verifiable instead of blind.
- **`backup/` is a safety net but not in git's job.** Phase 0.1 keeps it on disk, removes from history going forward.
- Pre-repair local snapshot archived at `/tmp/pickleball_backup_1780801472.tgz`.
