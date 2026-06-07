# ADR-0006: FastAPI for the API (supersedes ADR-0001)

**Status:** Accepted · **Date:** 2026-06-07 · **Supersedes:** ADR-0001

## Context
ADR-0001 chose Flask for a fast initial runnable app. Moving to the managed
production stack (ADR-0005), the API is I/O-bound at the edge (uploads, polling,
Supabase/Modal/Bedrock calls) and needs Pydantic validation, OpenAPI, and a clean
fit with realtime/SSE progress — all first-class in FastAPI.

## Decision
Build the production API on **FastAPI** (`app/`), with dependency-injected auth
(`Depends(get_current_user)` verifying Supabase JWTs) and Pydantic models. The
legacy Flask app (`src/api`) is retired. Route logic and ownership checks port
~1:1 from the Flask blueprints.

## Consequences
- ✅ Async I/O, Pydantic schemas, auto OpenAPI docs/SDK gen, DI for testability.
- ✅ Offline-testable via `TestClient` + dependency overrides (in-memory repo /
  fake Modal) — 36 tests pass with no cloud/GPU.
- ➖ Two frameworks transiently coexist until `src/api` is deleted.

## References
- `app/main.py`, `app/deps.py`, `app/routers/` · supersedes `docs/adr/0001-flask-over-fastapi.md`
