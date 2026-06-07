"""FastAPI dependencies — current user, settings, admin guard."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.security.jwt import InvalidToken, claims_to_user, verify_supabase_jwt


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


def get_current_user(
    authorization: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Verify the Supabase JWT and return the current user. 401 if missing/invalid."""
    token = _bearer(authorization)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")
    try:
        claims = verify_supabase_jwt(token, settings)
    except InvalidToken as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e
    return claims_to_user(claims)


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Guard admin-only routes."""
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user
