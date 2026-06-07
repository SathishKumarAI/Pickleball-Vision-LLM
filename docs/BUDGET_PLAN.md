# 💰 Budget Plan — Cloud LLM + Video Pipeline

> Cost model for the "upload video → annotated output + coaching" product.
> Decision: **drop direct OpenAI; use a managed cloud provider** (AWS Bedrock /
> Azure / Google Vertex). Default backend stays `rule` (free); cloud LLM is opt-in.
> Figures are representative list prices (verify current pricing before commit) —
> use them for *relative* comparison and order-of-magnitude planning.

---

## Unit assumptions (per 2-min, 720p clip)
- GPU work ≈ **90 s** (FRAME_SKIP=3, batched) → **0.025 GPU-hr**.
- LLM coaching ≈ **~20 rally calls + 1 summary**, ~500 tok in + 500 tok out each
  → **~10k input + ~10k output tokens / video**.
- Storage ≈ **30 MB in + 30 MB out = 60 MB / video**.

## LLM cost / video (10k in + 10k out tokens)

| Provider (model) | $/1M in | $/1M out | **$/video** | Notes |
|---|---|---|---|---|
| **AWS Bedrock — Claude 3 Haiku** | ~0.25 | ~1.25 | **~$0.015** | default; Claude quality, managed |
| AWS Bedrock — Claude 3.5 Sonnet | ~3.0 | ~15.0 | ~$0.18 | premium quality |
| Azure — GPT-4o-mini | ~0.15 | ~0.60 | ~$0.0075 | cheapest hosted GPT-class |
| Google Vertex — Gemini 1.5 Flash | ~0.075 | ~0.30 | **~$0.004** | cheapest overall |
| **`rule` backend (default)** | — | — | **$0.00** | no LLM call |

## GPU compute / video (~0.025 hr)

| Provider (instance) | $/hr (on-demand) | **$/video** | Spot ≈ |
|---|---|---|---|
| AWS g5.xlarge (A10G) | ~1.00 | ~$0.025 | ~$0.008 |
| Azure NCasT4_v3 (T4) | ~0.53 | ~$0.013 | ~$0.005 |
| GCP n1 + T4 | ~0.45 | ~$0.011 | ~$0.004 |

## Storage / egress (per video, S3-class)
- Store 60 MB → ~$0.0014 / month (@ $0.023/GB-mo) — **negligible with TTL cleanup**.
- Egress (annotated download 30 MB) → ~$0.0027 (@ $0.09/GB). Use signed URLs/CDN.

---

## Blended cost / video (recommended stack)
**Bedrock Claude Haiku + AWS g5 spot + S3 + TTL:**
`LLM ~$0.015 + GPU ~$0.008 + storage/egress ~$0.004 ≈ **~$0.03 / video**.`
With `rule` backend (no LLM): **~$0.012 / video** (GPU-dominated).

## Monthly scenarios

| Videos/mo | GPU (spot) | LLM (Haiku) | Storage+egress | **Total/mo** |
|---|---|---|---|---|
| 1,000 | ~$8 | ~$15 | ~$4 | **~$27** |
| 10,000 | ~$80 | ~$150 | ~$40 | **~$270** |
| 100,000 | ~$800 | ~$1,500 | ~$400 | **~$2,700** |

> GPU on-demand (no spot) ≈ 3× the GPU line. LLM line → $0 if you ship `rule`-only.

---

## Recommendation
- **Default cloud provider: AWS Bedrock + Claude 3 Haiku.** Managed (no GPU to run
  for the LLM), Claude-quality coaching, pay-per-token, same account as GPU/S3.
- **Cheapest: Vertex Gemini Flash** (~3–4× cheaper LLM) — pick if cost > model choice.
- **Already on Azure?** Azure GPT-4o-mini is the path of least resistance.
- Keep **`rule` as default + fallback** so a provider outage/cost spike never breaks
  the product (already wired, P0-5).

## Cost controls (already designed in / cheap to add)
- `rule` default + LLM timeout fallback (P0-5) → bounded LLM spend.
- Cheapest tier models (Haiku / Flash / 4o-mini), cap `max_tokens=200`.
- FRAME_SKIP + 720p cap → bounded GPU seconds (P0-2 enforces input size).
- **Spot/preemptible GPU** for the worker pool → ~3× cheaper compute.
- **S3 lifecycle TTL** (P1-5) → storage doesn't grow unbounded.
- **Content-hash idempotency** (P1-3) → re-uploads skip GPU + LLM entirely.
- Per-user quota / rate-limit to cap a single account's spend.

## Free / dev tier
Local everything: `hf` (local Mistral/LLaMA) + local GPU + local disk = **$0 external**.
Good for dev and self-hosting; cloud is for managed scale.
