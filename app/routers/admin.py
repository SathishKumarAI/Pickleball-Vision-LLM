"""Admin — cross-user job inspection + application log viewer (admin gated)."""

from typing import Optional

from fastapi import APIRouter, Depends

from app.deps import get_repo, require_admin
from app.logging_config import get_ring
from app.services.repo import Repo

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/jobs")
def all_jobs(limit: int = 100, _admin=Depends(require_admin), repo: Repo = Depends(get_repo)):
    """List all jobs across users (admin only)."""
    jobs = repo.list_all_jobs(limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/logs")
def app_logs(limit: int = 200, level: Optional[str] = None, _admin=Depends(require_admin)):
    """Recent application logs from the in-memory ring buffer (newest first).

    For durable/aggregated logs ship JSON logs (LOG_JSON=true) to your log
    platform; this endpoint is the live in-app viewer for quick troubleshooting.
    """
    records = get_ring().records(limit=limit, level=level)
    return {"logs": records, "count": len(records)}
