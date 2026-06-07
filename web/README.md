# Pickleball Vision — Web (Next.js)

Customer-facing app: auth, video upload, live job progress, annotated-video player,
coaching report, history, billing, settings. Talks to the FastAPI control plane
(`app/`) and Supabase directly.

## Stack
- Next.js 14 (App Router) + TypeScript + Tailwind
- `@supabase/ssr` for auth + realtime (live job progress via `postgres_changes`)
- Uploads go **direct to Supabase Storage** (RLS-gated); the API never streams video.

## Setup
```bash
cd web
cp .env.example .env.local   # fill Supabase URL/anon key + API URL
npm install
npm run dev                  # http://localhost:3000
```

## Structure
- `app/` — App Router: landing, `/login`, and the authed `(app)` group
  (dashboard, upload, jobs/[id] live progress, analyses/[id], history, billing, settings).
- `components/` — Nav, UploadDropzone, JobProgress, QuotaMeter, CoachingReport.
- `lib/` — `supabase/{client,server}`, `api.ts` (JWT-attaching fetch), `hooks/useJobRealtime`.
- `middleware.ts` — refreshes the Supabase session cookie.

## Deploy
Vercel: set `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`,
`NEXT_PUBLIC_API_URL`. `npm run build`.
