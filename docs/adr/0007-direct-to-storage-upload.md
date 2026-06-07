# ADR-0007: Browser uploads directly to Supabase Storage

**Status:** Accepted · **Date:** 2026-06-07

## Context
Match videos are large (up to ~300 MB). Streaming them through the FastAPI
control plane (the original Flask `f.save()` path) makes the API stateful,
memory-heavy, and a throughput bottleneck — bad for a stateless, cheap API tier.

## Decision
The **browser uploads directly to Supabase Storage** (RLS-gated to the user's own
`uid/` folder), then calls `POST /jobs` with the resulting object key. The Modal
worker pulls the object; outputs are written back to Storage and served to the
client via signed download URLs. The API never streams video bytes.

## Consequences
- ✅ API stays stateless, light, and cheap; no large-body handling.
- ✅ Offloads bandwidth to Supabase/CDN; resumable uploads possible.
- ➖ Upload trust boundary shifts to the client → **authoritative
  `validate_upload` re-check inside the worker** + RLS own-folder policy + size
  cap. (Implemented.)

## References
- `web/components/UploadDropzone.tsx` · `supabase/migrations/0002_storage.sql`
- `app/routers/jobs.py` · `src/api/validate.py`
- [Supabase Storage uploads](https://supabase.com/blog/storage-v3-resumable-uploads)
