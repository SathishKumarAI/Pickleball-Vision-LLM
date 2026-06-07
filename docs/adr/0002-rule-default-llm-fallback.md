# ADR-0002: `rule` is the default LLM backend and the universal fallback

**Status:** Accepted · **Date:** 2026-06-07

## Context
Coaching feedback can come from a heuristic (`rule`), a local model (`hf`), or a
managed cloud LLM (`cloud`). The pipeline has a hard ~2-minute latency budget. An
external/model call inside that budget is a variable-latency, possibly-failing,
possibly-costly dependency. We must never let the coaching step blow the budget or
fail the whole job.

## Decision
- **Default backend = `rule`** everywhere (Pipeline, jobs).
- `rule` is also the **timeout/error fallback** for `hf`/`cloud`: model calls run
  under a wall-clock `deadline_s`; on timeout or any error, return the rule coach
  (`CoachingFeedbackGenerator(fallback=True)`).

## Consequences
- ✅ MVP ships with zero LLM cost/latency/provider dependency.
- ✅ Bounded latency + graceful degradation; a slow/failing LLM can't break a job.
- ✅ `rule` runs with no ML deps → the whole pipeline is testable offline.
- ➖ `rule` advice is generic; real model coaching is a later, opt-in upgrade.

## References
- `src/llm/generate_feedback.py` · `docs/REMEDIATION_PLAN.md` (P0-5)
