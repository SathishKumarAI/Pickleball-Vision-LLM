# RFC-003: Managed Production Stack (supersedes RFC-002)

| | |
|---|---|
| **Status** | Accepted (implemented) |
| **Author** | Sathish Kumar |
| **Reviewers** | TBD |
| **Created / Updated** | 2026-06-07 |
| **Supersedes** | RFC-002 (AWS ECS/Celery/Terraform) |

## Summary
The product targets **~200 active customers** with the owner explicitly wanting
**minimal future maintenance**. At that scale the RFC-002 self-managed AWS stack
(ECS-EC2 Spot + Celery + ASG + Terraform) is over-built. This RFC adopts a
**managed-first stack** — FastAPI + **Modal** (serverless GPU) + **Supabase**
(Postgres/auth/storage/realtime) + **Next.js/Vercel** + **Stripe** + **Bedrock**
Claude Haiku — so almost nothing is self-operated. This stack is **built and
offline-verified** (see the repo: `app/`, `worker/`, `web/`, `supabase/`).

## Context & problem statement
RFC-001 defines the pipeline; RFC-002 defined heavy AWS infra for 10k+/mo. For
200 customers (a handful of concurrent jobs, bursty/mostly-idle), the operational
cost of running Kubernetes-adjacent infra dwarfs the compute. We want managed
services that scale to zero and require no cluster/queue/state ops.

## Goals
- Production-grade product (all features) with near-zero infra maintenance.
- Scale-to-zero GPU; durable state; realtime UX; billing; GDPR.
- ~80% of the existing Python core reused unchanged.

## Non-Goals
- Hyperscale (>100k/mo) — revert toward RFC-002 only if scale demands it.
- Multi-cloud, Kubernetes, self-hosted queue/DB.

## Proposed design (as built)
```
Next.js/Vercel ──Supabase Auth JWT──▶ FastAPI (app/, stateless)
  ├─ upload direct ▶ Supabase Storage (RLS)      (API never streams video)
  └─ realtime subscribe jobs row ◀── progress
FastAPI: verify JWT (JWKS/HS256) · quota · insert jobs row · Modal.spawn()
   ▼
Modal serverless GPU (worker/): run_job → Pipeline.process_video (detect→track→
   homography→analytics→annotate→encode) → progress PATCH → outputs to Storage →
   usage++  · Bedrock Haiku coaching (rule fallback)
   ▼
Supabase Postgres (profiles/jobs/analyses/subscriptions/usage, RLS) · Stripe
```
Components (all implemented):
- **`app/`** — FastAPI: auth (Supabase JWT `Depends`), jobs (upload-url/create/
  list/get/cancel/result/video/quota), billing (Stripe checkout/portal/webhook),
  account (usage + GDPR delete), admin. Repo/Dispatch/Storage are interfaces with
  in-memory/fake impls for offline tests and Supabase/Modal impls for prod.
- **`worker/`** — `runner.run_job` (pure, tested) + `modal_app.py` (GPU function);
  idempotency via atomic `claim_running` + `content_sha256`; progress → Supabase.
- **`web/`** — Next.js App Router; live progress via Supabase realtime; direct
  storage upload; Stripe checkout; GDPR delete.
- **`supabase/migrations/`** — schema + RLS + storage policies.

## Alternatives considered
- **RFC-002 AWS ECS/Celery/Terraform** — correct at 10k+/mo, too heavy at 200
  customers; rejected (this RFC supersedes it).
- **Replicate / RunPod** instead of Modal — comparable serverless GPU; Modal chosen
  for first-class Python + custom-pipeline flexibility + memory snapshotting.
- **Cognito/Clerk + RDS/Neon** instead of Supabase — viable; Supabase chosen to
  collapse auth+DB+storage+realtime into one managed platform (least ops).
- **Keep Flask** — replaced by FastAPI for async I/O, Pydantic, OpenAPI, and SSE/
  realtime fit (see ADR-0006).

## Trade-offs & risks
- **Modal cold starts** (bursty traffic) → weights Volume + memory snapshot +
  optional keep-warm; measure in the Phase-1 seam run on a GPU.
- **Vendor lock-in** (Modal/Supabase) — accepted for the maintenance savings; the
  Python core stays portable; Repo/Dispatch/Storage interfaces ease a swap.
- **Direct-to-storage upload** trust boundary → authoritative `validate_upload`
  re-check in the worker + RLS own-folder policy + size cap.
- **The GPU seam is still unproven end-to-end** (offline-tested with fakes) — the
  one remaining must-do is a real Modal run on a GPU (RFC-001 M0).

## Security / privacy / cost
- Supabase RLS (own-rows), service-role only server-side, signed download URLs,
  Stripe-signed webhooks, GDPR delete + storage TTL.
- **Cost @ ~200 customers (rule-coaching MVP):** Modal GPU pay-per-second
  (scale-to-zero, ~$0.01–0.03/video), Supabase free/Pro (~$0–25/mo), Vercel
  hobby/Pro (~$0–20/mo), Stripe % of revenue, Bedrock only if `cloud` backend on.
  Order-of-magnitude **tens of dollars/month** at low volume — far below RFC-002's
  fixed infra. Verify with the BUDGET doc once the seam run gives real per-video GPU seconds.

## Rollout & testing plan
- Offline (done): 36 pytest pass across API/jobs/worker/billing/account/admin/
  analytics; Next.js `build` clean.
- Remaining: provision Supabase + Modal + Vercel; run the **M0 seam** (one real
  clip through `/jobs` → Modal → Storage → UI); then staged launch.

## References / Further reading
- Modal: [modal.com](https://modal.com/) · [serverless GPUs / snapshot](https://modal.com/blog/truly-serverless-gpus) · [web endpoints](https://modal.com/docs/guide/webhooks)
- Supabase: [Auth JWTs](https://supabase.com/docs/guides/auth/jwts) · [RLS](https://supabase.com/docs/guides/database/postgres/row-level-security) · [FastAPI + Supabase](https://dev.to/zwx00/validating-a-supabase-jwt-locally-with-python-and-fastapi-59jf) · [Next.js + Supabase](https://supabase.com/docs/guides/getting-started/quickstarts/nextjs) · [Storage uploads](https://supabase.com/blog/storage-v3-resumable-uploads)
- IaC trade-off (why managed): [Spacelift CDK vs Terraform](https://spacelift.io/blog/aws-cdk-vs-terraform)
- Bedrock: [Claude on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- Stripe: [Checkout](https://stripe.com/docs/payments/checkout) · [Webhooks](https://stripe.com/docs/webhooks)
- Internal: `docs/specs/RFC-001`, `docs/specs/RFC-002` (superseded), `docs/ROADMAP.md`, `docs/adr/`
