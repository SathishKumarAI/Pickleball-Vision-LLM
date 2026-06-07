-- Pickleball-Vision-LLM — initial schema (Supabase Postgres)
-- Auth users + passwords are managed by Supabase in auth.users; we add app tables.

-- ---------------------------------------------------------------------------
-- profiles: 1:1 with auth.users
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
  id                 uuid primary key references auth.users(id) on delete cascade,
  email              text,
  full_name          text,
  role               text not null default 'user',          -- 'user' | 'admin'
  stripe_customer_id text,
  created_at         timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- jobs: control-plane row (replaces in-process JobStore)
-- ---------------------------------------------------------------------------
create table if not exists public.jobs (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references public.profiles(id) on delete cascade,
  status             text not null default 'queued',         -- queued|running|done|error|cancelling|cancelled
  progress           real not null default 0,
  message            text,
  input_object_key   text,
  content_sha256     text,                                   -- idempotency / dedup
  output_video_key   text,
  result_object_key  text,
  modal_call_id      text,
  error              text,
  duration_s         real,
  frames_processed   int,
  tracker            text default 'supervision',
  feedback_backend   text default 'rule',
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);
create index if not exists jobs_user_created_idx on public.jobs(user_id, created_at desc);
create index if not exists jobs_sha_idx on public.jobs(content_sha256);

-- ---------------------------------------------------------------------------
-- analyses: structured insight payload (one per finished job)
-- ---------------------------------------------------------------------------
create table if not exists public.analyses (
  id          uuid primary key default gen_random_uuid(),
  job_id      uuid not null references public.jobs(id) on delete cascade,
  user_id     uuid not null references public.profiles(id) on delete cascade,
  summary     text,
  states      jsonb,            -- per-frame GameState[]
  metrics     jsonb,            -- kitchen usage, rally tempo, heatmap bins, shot counts
  homography  jsonb,            -- court calibration (Phase 6)
  players     jsonb,            -- per-player profiles (Phase 6 ReID)
  created_at  timestamptz not null default now()
);
create index if not exists analyses_user_idx on public.analyses(user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- subscriptions: Stripe state mirror
-- ---------------------------------------------------------------------------
create table if not exists public.subscriptions (
  user_id              uuid primary key references public.profiles(id) on delete cascade,
  stripe_subscription_id text,
  plan                 text not null default 'free',         -- free|starter|pro
  status               text not null default 'active',
  current_period_end   timestamptz,
  updated_at           timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- usage: monthly metering for quota
-- ---------------------------------------------------------------------------
create table if not exists public.usage (
  user_id           uuid not null references public.profiles(id) on delete cascade,
  period            text not null,                           -- 'YYYY-MM'
  videos_processed  int not null default 0,
  seconds_processed int not null default 0,
  primary key (user_id, period)
);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------
alter table public.profiles      enable row level security;
alter table public.jobs          enable row level security;
alter table public.analyses      enable row level security;
alter table public.subscriptions enable row level security;
alter table public.usage         enable row level security;

-- Users see only their own rows. The service-role key (API + Modal) bypasses RLS.
create policy "own profile"        on public.profiles      for select using (id = auth.uid());
create policy "own jobs"           on public.jobs          for select using (user_id = auth.uid());
create policy "own analyses"       on public.analyses      for select using (user_id = auth.uid());
create policy "own subscription"   on public.subscriptions for select using (user_id = auth.uid());
create policy "own usage"          on public.usage         for select using (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- On new auth user: create a profile + a free subscription
-- ---------------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email, full_name)
    values (new.id, new.email, coalesce(new.raw_user_meta_data->>'full_name', ''));
  insert into public.subscriptions (user_id, plan, status)
    values (new.id, 'free', 'active');
  return new;
end; $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
