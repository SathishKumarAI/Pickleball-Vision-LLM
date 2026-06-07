# ADR-0003: Managed cloud LLM (AWS Bedrock), not direct OpenAI

**Status:** Accepted · **Date:** 2026-06-07

## Context
Coaching feedback can use a hosted LLM. A direct OpenAI dependency means a paid
closed API, per-call latency inside a tight budget, and shipping frames/captions
of people's faces to a third party (privacy/GDPR). The user directed: use a
managed cloud provider instead of OpenAI.

## Decision
Use a **managed cloud provider** for the `cloud` backend — **AWS Bedrock + Claude
3 Haiku** (default), with Azure / Vertex as alternates. Remove the direct OpenAI
backend (kept commented for reference). Default backend stays `rule`; canonical
real backend is local `hf` (see ADR-0002).

## Consequences
- ✅ Managed, pay-per-token, same cloud as the rest; Claude-quality coaching.
- ✅ No direct OpenAI dependency; data stays within the chosen provider.
- ➖ Provider-specific SDKs (lazy-imported via `[bedrock]`/`[azure]`/`[vertex]`).
- 🔒 Default `rule` + timeout fallback means an outage never breaks the product.

## References
- `src/llm/generate_feedback.py` · `docs/BUDGET_PLAN.md`
- [Claude on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
