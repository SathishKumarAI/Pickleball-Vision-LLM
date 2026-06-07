# 🚀 Getting Started — Run & Test the Product

> Everything required to run and test Pickleball-Vision-LLM, from "nothing
> installed" to "full cloud product". Two tracks: **offline** (no cloud/GPU, ~80%
> of the product) and **cloud** (Modal + Supabase + Vercel for real inference).

---

## 1. Prerequisites (tools)
| Tool | Why | Min version |
|------|-----|-------------|
| Python | API + worker + tests | 3.10–3.12 |
| Node + npm | Next.js frontend | Node 18+ |
| ffmpeg / ffprobe | upload validation + decode | any recent |
| git | repo | any |
| (cloud) Modal CLI | serverless GPU | latest |
| (cloud) Supabase CLI | DB/migrations | latest |

## 2. Accounts needed (cloud track only)
| Account | Used for | Cost to start |
|---------|----------|---------------|
| **Supabase** | Postgres + Auth + Storage + Realtime | Free tier |
| **Modal** | serverless GPU inference | Free credits |
| **Vercel** | frontend hosting | Hobby free |
| **AWS (Bedrock)** | LLM coaching (optional; `rule` default) | pay-per-token |
| **Stripe** | billing (optional for testing) | Free test mode |

> You can do the **entire offline track with none of these.**

---

## 3. Offline track (no cloud/GPU) — start here
Verifies the API, jobs control plane, billing logic, analytics, logging, and the
frontend UI. Real video inference is the only thing not exercised.

```bash
# backend env
python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings pyjwt httpx \
            python-multipart numpy pytest

# run the API (in-memory repo + fake Modal + rule coaching)
PYTHONPATH=. uvicorn app.main:app --reload          # http://localhost:8000/health

# run the full offline test suite (42 tests)
PYTHONPATH=. pytest -q

# run the frontend
cd web && cp .env.example .env.local && npm install && npm run dev   # http://localhost:3000
```
**What works offline:** landing/login UI, `/analyze` (fusion+coaching+analytics),
jobs state machine (fake spawn), quota/billing logic, admin logs viewer, court
analytics. **What doesn't:** real YOLO/track/annotate (needs GPU), live Supabase
auth/realtime (needs a Supabase project).

### Try the API offline
```bash
# mint a dev JWT (HS256 with the default dev secret) and call /analyze
python - <<'PY'
import jwt, requests
tok = jwt.encode({"sub":"u1","email":"a@b.com","aud":"authenticated",
                  "app_metadata":{"role":"admin"}}, "dev-insecure-change-me", algorithm="HS256")
r = requests.post("http://localhost:8000/analyze",
    headers={"Authorization":f"Bearer {tok}"},
    json={"detections":[[{"bbox":[0,0,9,9],"confidence":0.9,"class_id":32,"class_name":"sports ball"}]]})
print(r.status_code, r.json()["summary"])
PY
```

---

## 4. Cloud track (real product)
### 4.1 Supabase
```bash
# create a project at supabase.com, then:
supabase link --project-ref YOUR_REF
supabase db push                      # applies supabase/migrations/*.sql (schema + RLS + storage)
```
Grab from Settings → API: `SUPABASE_URL`, anon key, **service-role key**, JWT secret.

### 4.2 Backend env (`.env` at repo root)
```dotenv
SUPABASE_URL=https://YOUR.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_KEY=...            # server-only
SUPABASE_JWT_SECRET=...             # Settings → Auth
APP_SECRET=<random-strong>
ENVIRONMENT=prod
# optional
AWS_REGION=us-east-1               # Bedrock (cloud coaching)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_STARTER=price_...
STRIPE_PRICE_PRO=price_...
SENTRY_DSN=                         # optional error tracking
LOG_JSON=true                       # prod JSON logs
```

### 4.3 Modal (GPU worker)
```bash
pip install modal && modal token new
modal secret create pvllm-secrets SUPABASE_URL=... SUPABASE_SERVICE_KEY=... AWS_REGION=us-east-1
modal deploy worker/modal_app.py          # deploys run_analysis + retention cron
```
Upload a detector weight to the `pvllm-weights` Volume (a Roboflow pickleball YOLO
`.pt`; see `docs/MODELS_AND_REUSE.md`) and set `DETECTOR_MODEL`.

### 4.4 Frontend (Vercel)
Set `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL`;
deploy `web/`.

### 4.5 Seed demo users + sign in
```bash
SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/seed_demo_users.py
```
Log in with the creds in `docs/DEMO_ACCESS.md` (admin sees the **Admin** log viewer).

---

## 5. The one must-do before launch — M0 seam run 🖥️
The control plane is offline-tested with fakes; the **real** GPU path has not run
end-to-end. Run one real clip through `POST /jobs` → Modal → Storage → `GET
/jobs/{id}/video`, and record per-stage timings + GPU memory + cold-start
(`docs/thinking/M0-seam-results.md`). Everything downstream assumes this works.

## 6. Test matrix
| Layer | How to test | Needs |
|-------|-------------|-------|
| API / jobs / billing / auth | `pytest tests/` | offline |
| Analytics / homography | `pytest tests/test_analytics.py` | offline |
| Logging / admin viewer | `pytest tests/test_logging_admin.py` | offline |
| Frontend build/types | `cd web && npm run build` | offline |
| Auth / RLS / realtime / storage | manual on a Supabase project | Supabase |
| Real inference (M0) | `modal run worker/modal_app.py` + a clip | Modal + GPU |

## References
- `docs/INFRA_SCALING.md` (min→max infra) · `docs/DEMO_ACCESS.md` (logins + logs) ·
  `docs/specs/RFC-003-managed-stack.md` (architecture) · `docs/TASKS.md` (status).
