"""FastAPI application factory.

Replaces the Flask `create_app` (`src/api/__init__.py`). Routers are added as the
product comes online (Phase 2+: jobs, billing, account, admin).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze, auth, health


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Pickleball-Vision-LLM API",
        version="0.2.0",
        description="Control plane: auth, jobs, billing. GPU work runs on Modal.",
    )
    origins = (
        ["*"] if settings.cors_origins.strip() == "*"
        else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(analyze.router)
    # Phase 2+: jobs, analyses, billing, account, admin
    return app


app = create_app()
