# 🏗️ Infrastructure: Minimum → Maximum (DevOps · Compute · Cost)

> How to run the product at every scale, from a system-architecture view. The
> codebase is built for the **managed stack** (RFC-003); the self-managed AWS path
> (RFC-002) is the high-scale option. Costs are order-of-magnitude (verify live
> pricing). "DevOps effort" = ongoing human ops, not one-time setup.

The product has 4 planes that scale independently:
- **Frontend** (Next.js) — static/SSR, cheap, scales trivially.
- **API control plane** (FastAPI) — stateless, CPU-only, light.
- **GPU worker** (vision inference) — the expensive, scale-driving plane.
- **Data** (Postgres + object storage + auth + realtime).

---

## Tier 0 — Minimum (dev / demo) · ~$0/mo
**Who:** one developer, no real GPU inference.
- **Compute:** your laptop. FastAPI (`uvicorn app.main:app`) with the in-memory
  repo + fake dispatcher + `rule` coaching; Next.js `npm run dev`. The no-GPU
  `/analyze` + analytics paths run fully (that's the 36-test offline surface).
- **Data:** in-memory / local SQLite (legacy) or a free Supabase project.
- **GPU:** none — real `process_video` is stubbed/skipped.
- **DevOps effort:** none. **Cost:** $0.
- **System arch:** single process; no queue, no scale-out. Good for building +
  demoing the control plane and analytics, not for processing real videos.
- **Run:** `uvicorn app.main:app --reload` + `cd web && npm run dev`.

## Tier 1 — Minimum *production* (launch / ~200 customers) · ~$0–60/mo  ← **target**
**Who:** the real product at low, bursty volume. **No servers to run.**
- **Compute:** **Modal** serverless GPU (scale-to-zero, pay-per-second, 1 A10G/T4
  per job, `max_containers` caps concurrency); **Vercel** for Next.js; FastAPI on
  a small always-on host (Fly.io / Render / Railway / Modal web endpoint).
- **Data:** **Supabase** (Postgres + Auth + Storage + Realtime), Free → Pro.
- **LLM:** `rule` (free) default; **Bedrock Haiku** opt-in.
- **DevOps effort:** ~near-zero — all managed, scale-to-zero, no cluster/queue/DB ops.
- **Cost:** Modal ~$0.01–0.03/video (≈$5–20/mo at hundreds of videos), Supabase
  $0–25, Vercel $0–20, Stripe = % of revenue → **tens of $/mo**.
- **System arch:** browser → Supabase (auth/storage/realtime) + FastAPI → `Modal.spawn`
  → GPU function → Supabase. No queue to manage (Modal queues internally); no DB to
  run (Supabase). This is RFC-003 / [ADR-0005](adr/0005-managed-stack.md).
- **Run:** `modal deploy worker/modal_app.py` · deploy `app/` to Fly/Render · deploy
  `web/` to Vercel · `supabase db push` migrations. Set env (Supabase keys, Modal
  token, Bedrock creds, Stripe keys).

## Tier 2 — Growth (~1k–10k videos/mo) · ~$100–500/mo
**Who:** steady paying base.
- **Compute:** same managed stack; Modal autoscales (raise `max_containers`,
  add `min_containers`/keep-warm during peaks to cut cold starts). FastAPI 2–3
  instances behind the platform LB.
- **Data:** Supabase Pro + compute add-on / read replica; Storage lifecycle TTL
  (retention cron already in `worker/modal_app.py`).
- **Observability:** turn on Sentry (`SENTRY_DSN`) + Supabase/Modal dashboards;
  watch p95 latency, GPU seconds, queue depth, error rate.
- **DevOps effort:** light (part-time) — mostly dashboards + cost tuning.
- **Cost:** Modal dominates (~$0.02/video × volume), Supabase ~$25–100, Vercel
  ~$20–150 → **low hundreds $/mo**.
- **System arch:** unchanged shape; tune knobs (FRAME_SKIP, batch, keep-warm,
  cheapest GPU that fits). **Decision point:** when steady GPU spend approaches a
  full-time GPU's cost, evaluate Tier 3.

## Tier 3 — Maximum / high-scale (100k+/mo) · ~$1k–10k+/mo
**Who:** large, sustained load where pay-per-second GPU > owning GPUs.
- **Compute:** the self-managed **AWS** design (RFC-002): **ECS-on-EC2 g5 Spot**
  GPU pool (Capacity Provider, autoscale on queue depth, DCGM metrics), **Celery**
  on **ElastiCache Redis**, **ECS Fargate** API behind **ALB**, **TensorRT/ONNX**
  exported models, **NVENC** hardware encode.
- **Data:** **RDS Postgres** (Multi-AZ) + **S3** + **CloudFront**; or keep Supabase
  if it scales for you.
- **DevOps effort:** real — dedicated platform/DevOps owner; Terraform/CDK IaC,
  on-call, capacity planning, Spot-interruption handling.
- **Cost:** GPU fleet (Spot) + Redis + RDS + S3 + transfer → **thousands $/mo**
  (cheaper *per video* than Modal at this scale, more expensive to operate).
- **System arch:** RFC-002 topology. Hybrid is valid — keep Modal for burst
  overflow, own baseline GPUs for steady load.

---

## Compute sizing cheat-sheet
| Driver | Lever |
|--------|-------|
| Per-video GPU seconds | `FRAME_SKIP` (3), 720p cap, GPU batch 16–32, NVENC encode |
| Concurrency | Modal `max_containers` / Celery worker concurrency = real GPU slots |
| Cold start | weights Volume + Modal memory snapshot + `min_containers` keep-warm |
| LLM cost | `rule` default; Bedrock Haiku opt-in; `max_tokens` cap; timeout fallback |
| Storage growth | retention cron + Supabase/S3 lifecycle TTL |
| Idempotency / waste | content-sha256 dedup (re-uploads skip the GPU) |

## DevOps effort by tier
Tier 0–1: **near-zero** (managed, scale-to-zero). Tier 2: **part-time**
(dashboards, cost tuning). Tier 3: **dedicated** (IaC, on-call, capacity).

## How to choose
Start at **Tier 1** (where the code is today). Stay managed through Tier 2.
Move to **Tier 3 only when** sustained GPU spend on Modal clearly exceeds the
all-in cost (incl. a DevOps salary) of owning GPUs — typically well past 10k/mo.

## References / Further reading
- Managed: [Modal pricing/GPU](https://modal.com/pricing) · [Supabase pricing](https://supabase.com/pricing) · [Vercel pricing](https://vercel.com/pricing)
- High-scale AWS: `docs/specs/RFC-002-aws-infrastructure.md`, `docs/BUDGET_PLAN.md`
- Current design: `docs/specs/RFC-003-managed-stack.md` · [ADR-0005](adr/0005-managed-stack.md)
