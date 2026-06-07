# 🗺️ Master Roadmap — Pickleball-Vision-LLM

> Single source of truth tying every plan together. Updated 2026-06-07.
> Product: web app — user logs in, uploads a pickleball game video, gets an
> annotated output video + coaching insights back in minutes.

---

## 1. Where we are (state snapshot)

**Done + verified offline (no ML deps):**
- Repo repaired (flattened scaffold), synced to real GitHub `main`, packaged
  (`pyproject`, `src` layout, `pickleball` entrypoint), runnable Flask app.
- De-duplicated `src/`; vision layer **import-coherent** (17 broken imports → 0).
- Pipeline glue: `GameStateBuilder` (fusion) → `CoachingFeedbackGenerator` →
  `Pipeline.analyze_detections`/`process_video` (lazy vision, annotated-mp4 output).
- Product control plane: async **JobStore** + worker, `/jobs/video` upload →
  poll → download, **auth** (register/login/me, bearer tokens) + per-user
  ownership, `GET /jobs` listing.
- P0 hardening: GPU concurrency gate, ffprobe input validation, timeout + cancel,
  LLM timeout/error fallback to `rule`.
- LLM: `rule` (default) / `hf` (local) / `cloud` (Bedrock·Azure·Vertex). OpenAI
  dropped (commented).
- System design **image** rendered (`docs/assets/system_design.png`).

**Not yet real (needs GPU box / infra):**
- Actual model inference (YOLO/track/pose/annotate/encode) never run end-to-end.
- Durable state (Redis), task queue (Celery), object storage (S3), observability.
- Vision correctness (court homography, action model, re-ID, multi-person pose).

> ⚠️ **#1 risk:** the control plane was tested with a **stub processor**. The seam
> to the real GPU pipeline (file handoff, progress timing, memory, true encode
> cost) has never run together. First milestone must close this.

---

## 2. Plan document map

| Doc | Scope |
|---|---|
| `docs/ROADMAP.md` (this) | master plan, milestones, priorities |
| `docs/PLAN.md` | engineering phases 0–5 (repair → runnable → dedup → wire → tests → deploy) |
| `docs/REMEDIATION_PLAN.md` | production-hardening P0/P1/P2 (from arch review) |
| `docs/thinking/mini-features-backlog.md` | 20 mini-features, 7 epics, sub-tasks |
| `docs/thinking/video-product-architecture.md` | upload→annotated-video design + latency budget |
| `docs/ARCHITECTURE.md` | system diagram + tech stack |
| `docs/assets/system_design.png/.svg` | product architecture image |
| `docs/MODELS_AND_REUSE.md` | OSS models/repos to reuse (Roboflow, TrackNet, supervision) |
| `docs/BUDGET_PLAN.md` | cloud LLM + GPU + storage cost model |
| `docs/SETUP.md` | install / run / config |
| `docs/thinking/*-iter*.md` | per-iteration decision notes |

---

## 3. Delivery milestones (exit criteria)

### M0 — Prove the seam (🖥️ GPU box) · BLOCKER
Wire the stub control plane to the real `Pipeline.process_video` end-to-end.
- **Exit:** one real clip → annotated mp4 + insights via `/jobs/video`; measured
  per-stage timings; confirmed model memory co-fit; encode cost known.

### M1 — Single-box MVP, production-safe
P0 items (mostly done) + the seam, deployable on one GPU box.
- **Exit:** auth + upload + bounded GPU + input validation + timeout/cancel +
  LLM fallback all working on real videos; Docker image builds & serves.

### M2 — Reliability & scale-out
Durable + horizontally scalable.
- **Exit:** Redis JobStore, Celery GPU queue (acks_late, soft/hard limits), S3
  output + signed URLs, autoscaling worker pool, retries + content-hash idempotency.

### M3 — Observability & lifecycle
Operable against the latency SLA.
- **Exit:** p95 + per-stage latency, GPU/queue metrics, alerts; data TTL/GC,
  `DELETE /jobs/<id>`, retention policy; auth rate-limit + token revocation.

### M4 — Vision correctness (coaching gets good)
- **Exit:** court homography (real-unit positions), action classifier (named
  model), per-player re-ID, multi-person pose, ball-every-frame; pickleball
  weights (Roboflow) in use.

### M5 — Product UX
- **Exit:** web dashboard (upload → annotated video + states + heatmaps),
  SSE/websocket progress, completion webhook/email, analytics (heatmaps, tempo).

---

## 4. Prioritized path (next, in order)

1. **M0 seam test** (🖥️) — nothing else is trustworthy until this runs.
2. **G1 test suite + CI green** (💻) — lock in the verified behavior as regression
   tests; fix the stale `ci-cd.yml`. *(in progress: `tests/conftest.py` added.)*
3. **P0-3 Redis JobStore** (🧰) — durability; unblocks M2.
4. **P1-1/P1-2 S3 + Celery** (🧰) — real scale-out; removes the in-process-thread
   root cause.
5. **P1-4 observability** (🖥️) — you sell a latency SLA; measure it.
6. **M4 vision correctness** (🖥️) — court homography first (biggest coaching gap).

Legend: 💻 verifiable on this machine · 🖥️ needs GPU box · 🧰 needs infra (Redis/S3/Celery).

---

## 5. Decision log

| Decision | Why |
|---|---|
| **Flask** (not FastAPI) | entry point already Flask-shaped; FastAPI relic was broken |
| **`rule` LLM default + fallback** | zero-dep, can't blow latency budget (P0-5) |
| **Local `hf` canonical real LLM** | OSS-first, on-box, no per-call cost |
| **Cloud = Bedrock/Azure/Vertex, not OpenAI** | user decision; Bedrock Claude Haiku default; see BUDGET_PLAN |
| **Reuse OSS** (supervision, Roboflow, TrackNet) | don't reinvent; see MODELS_AND_REUSE |
| **Lazy imports + dep-free fallback everywhere** | offline-verifiable; heavy stack only on GPU box |
| **GPU semaphore now, Celery queue later** | single-box bridge to real admission control |

---

## 6. Open questions

**Resolved 2026-06-07** → see `docs/DELIVERY_PLAN.md`:
- ✅ Cloud account: **AWS** (Bedrock + S3 + Celery/Redis + g5 GPU).
- ✅ Scale: **~10k videos/mo** (small autoscaling pool, Redis+Celery, S3).
- ✅ Coaching bar: **`rule`-only MVP first**; cloud LLM (Bedrock Haiku) post-MVP.

**Still open (smaller):**
- Data retention window for uploaded videos (faces → GDPR; sets S3 TTL).
- IaC: Terraform vs AWS CDK.
- Auth: custom bearer tokens (MVP) vs Cognito.
- Real-time progress: SSE vs websocket vs poll (MVP = poll).

---

## 7. Standing working agreement
- One branch/commit per feature + a `docs/thinking/` note per iteration.
- Reuse OSS before writing model code ([[prefer-oss-libs]]).
- Cloud LLM = managed provider, never direct OpenAI ([[cloud-llm-not-openai]]).
- Verify before claiming done; lazy-import heavy deps so the offline path stays green.
- Commit + push after each completed task.

---

## References / Further reading
- Planning practice: [Pragmatic Engineer — RFCs & Design Docs](https://blog.pragmaticengineer.com/rfcs-and-design-docs/) · [Product School — PRD template](https://productschool.com/blog/product-strategy/product-template-requirements-document-prd)
- Internal docs: `docs/DELIVERY_PLAN.md` · `docs/PLAN.md` · `docs/REMEDIATION_PLAN.md` · `docs/BUDGET_PLAN.md` · `docs/MODELS_AND_REUSE.md` · `docs/ARCHITECTURE.md` · `docs/specs/` (PRD, RFC-001, RESEARCH_NOTES) · `docs/thinking/`
