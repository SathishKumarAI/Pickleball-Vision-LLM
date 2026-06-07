"""Health + service metadata."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Liveness probe."""
    return {"status": "healthy"}


@router.get("/")
def index():
    """Service metadata."""
    return {
        "service": "pickleball-vision-llm",
        "version": "0.2.0",
        "stack": ["fastapi", "modal", "supabase", "nextjs", "bedrock", "stripe"],
        "endpoints": ["/health", "/auth/me", "/analyze", "/jobs", "/billing", "/account"],
    }
