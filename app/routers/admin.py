"""Admin — cross-user job inspection (admin-claim gated)."""

from fastapi import APIRouter, Depends

from app.deps import get_repo, require_admin
from app.services.repo import Repo

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/jobs")
def all_jobs(limit: int = 100, _admin=Depends(require_admin), repo: Repo = Depends(get_repo)):
    """List all jobs across users (admin only)."""
    jobs = repo.list_all_jobs(limit)
    return {"jobs": jobs, "count": len(jobs)}
