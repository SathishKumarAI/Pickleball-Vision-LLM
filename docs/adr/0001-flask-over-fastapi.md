# ADR-0001: Flask over FastAPI for the API

**Status:** Superseded by [ADR-0006](0006-fastapi-over-flask.md) · **Date:** 2026-06-07

> Superseded: the product migrated to FastAPI when moving to the managed
> production stack (async I/O, Pydantic, OpenAPI, realtime fit). This ADR records
> why Flask was the right *initial* call.

## Context
The repo had a broken FastAPI experiment under `src/temp/api` (imports referenced a
dead `pickleball_vision` layout). The real entry point `src/main.py` was written in
the Flask idiom: a `create_app()` factory + `app.run(debug=...)`. We needed a
runnable app fast, and the local env already had Flask installed (not FastAPI).

## Decision
Use **Flask** as the API framework. Build `src/api/create_app()` as the factory;
register feature blueprints (`analyze`, `auth`, `jobs`). Remove the broken FastAPI
relic (`src/temp`).

## Consequences
- ✅ App runs immediately; matches the existing entry point; minimal new deps.
- ✅ Blueprint structure scales to more endpoints cleanly.
- ➖ No built-in async I/O or automatic OpenAPI schema (FastAPI gives these).
- 🔁 Revisit if async I/O becomes dominant or we want first-class schema gen; both
  frameworks could coexist behind the same WSGI/ASGI server if needed.

## References
- [Flask app factories](https://flask.palletsprojects.com/en/latest/patterns/appfactories/)
- `docs/specs/RFC-001-video-analysis-pipeline.md` (Alternatives)
