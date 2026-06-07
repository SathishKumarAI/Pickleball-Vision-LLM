"""Application settings (Pydantic BaseSettings).

Single source of config for the API and the Modal worker. Values come from env
vars / a local ``.env``. No secrets are hard-coded; dev defaults are inert.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Supabase ---
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""           # server-only; bypasses RLS
    supabase_jwt_secret: str = "dev-insecure-change-me"  # HS256 verify (legacy keys)
    supabase_jwt_aud: str = "authenticated"
    supabase_jwks_url: str = ""              # set for RS256 signing-keys verification

    # --- Storage buckets ---
    uploads_bucket: str = "uploads"
    outputs_bucket: str = "outputs"

    # --- Modal ---
    modal_app_name: str = "pickleball-gpu"
    modal_function: str = "run_analysis"

    # --- LLM (AWS Bedrock; rule is the always-on default/fallback) ---
    feedback_backend: str = "rule"           # rule | hf | cloud
    llm_provider: str = "bedrock"            # bedrock | azure | vertex
    aws_region: str = "us-east-1"

    # --- Stripe ---
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # --- Quotas (videos / month per plan) ---
    free_monthly_videos: int = 3
    starter_monthly_videos: int = 50
    pro_monthly_videos: int = 500

    # --- Upload bounds (mirror src/api/validate.py) ---
    max_upload_mb: int = 300
    max_duration_s: int = 180

    # --- Misc ---
    cors_origins: str = "*"                  # comma-separated in prod
    environment: str = "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()
