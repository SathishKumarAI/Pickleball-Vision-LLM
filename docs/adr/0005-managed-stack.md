# ADR-0005: Managed stack (Modal + Supabase + Next.js) over self-managed AWS

**Status:** Accepted · **Date:** 2026-06-07

## Context
Target is **~200 active customers**, and the owner explicitly wants minimal
future maintenance. RFC-002's self-managed AWS design (ECS-EC2 Spot + Celery +
ASG autoscaling + Terraform) is built for 10k+/mo and is heavy to operate — a poor
fit at this scale (a handful of concurrent, bursty jobs).

## Decision
Adopt a **managed-first stack**:
- **Modal** — serverless GPU (scale-to-zero, pay-per-second); replaces ECS-EC2 +
  Celery + ASG + the GPU semaphore.
- **Supabase** — managed Postgres + Auth + Storage + Realtime; replaces RDS +
  custom auth + S3-wiring + a realtime layer, and the in-process JobStore/SQLite.
- **Next.js on Vercel** — managed frontend hosting.
- **Stripe** — billing. **Bedrock** — LLM (ADR-0003).
Supersedes RFC-002 (see RFC-003).

## Consequences
- ✅ Near-zero infra ops; scale-to-zero cost at low/bursty volume.
- ✅ ~80% of the Python core reused unchanged; control plane swapped to managed.
- ➖ Vendor lock-in (Modal/Supabase) — mitigated by keeping the portable Python
  core and Repo/Dispatch/Storage interfaces with swappable impls.
- 🔁 Revisit toward RFC-002 only if scale grows past ~10k/mo.

## References
- `docs/specs/RFC-003-managed-stack.md` (supersedes RFC-002) · `app/`, `worker/`, `web/`
- [Modal](https://modal.com/) · [Supabase](https://supabase.com/docs)
