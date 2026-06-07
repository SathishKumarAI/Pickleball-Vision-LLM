# Thinking — Video product + auth, iteration 1 result (2026-06-07)

## Built (💻 all verified offline: parse + dep-free smoke green)

### Async video-job control plane
- `src/services/jobs.py` — `Job`, `JobStore` (thread-safe), async `submit()` with
  injected processor + `progress_cb`. Lifecycle queued→running→done|error tested
  (sync, async, error path).
- `src/api/blueprints/jobs.py` — `POST /jobs/video` (upload→job_id, 202),
  `GET /jobs/<id>` status+progress, `/result`, `/video` download. Auth + ownership
  enforced.
- `src/pipeline.py` — `process_video(annotate_path=, progress_cb=)`: writes an
  annotated mp4 (supervision annotators, lazy; OpenCV fallback), holds last
  annotation on skipped frames for smooth output, streams progress.

### Auth / user DB (login system)
- `src/services/db.py` — `UserDB` (SQLite, stdlib), unique email, hashed pw.
- `src/services/auth.py` — pbkdf2 hashing (werkzeug) + signed expiring bearer
  tokens (itsdangerous; no PyJWT dep).
- `src/api/blueprints/auth.py` — `/auth/register|login|me` + `token_required`
  guard. `create_app` sets `SECRET_KEY` (APP_SECRET).
- Job routes now require auth; users only see their own jobs (403 otherwise).
- Verified: register 201, dup 409, short-pw 400, bad-login 401, me 200/401,
  jobs-no-auth 401, owner 200, other-user 403.

### System design image
- `docs/assets/system_design.svg` → `system_design.png` (rsvg-convert). Lanes:
  Client / API+Auth+UserDB / Worker→Pipeline stages / latency budget + scale-out.

## Latency strategy (the "couple minutes" ask)
FRAME_SKIP=3, GPU batch 16–32, 720p cap, async job + progress, hold-last
annotation. Target ≤90s for a 2-min clip on one GPU. (Real run = GPU box.)

## Deferred → 🖥️ GPU box
- Real `process_video` run (detect/track/annotate/encode) on the sample clip.
- hw-accelerated ffmpeg encode + GPU batching.
- thread worker → Celery/RQ + Redis JobStore + S3 object store for scale.
- DB: SQLite → Postgres/SQLAlchemy when multi-instance.

## Security TODO (before prod)
- Set strong `APP_SECRET`/`SECRET_KEY`; HTTPS only; rate-limit `/auth/*`;
  validate upload size/type; per-user storage quota; signed download URLs.
