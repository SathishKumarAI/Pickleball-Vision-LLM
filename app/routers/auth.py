"""Auth — current user. Register/login happen client-side via Supabase Auth."""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.deps import get_current_user
from app.models import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=UserOut)
def me(user: Dict[str, Any] = Depends(get_current_user)):
    """Return the authenticated user (from the verified Supabase JWT)."""
    return user
