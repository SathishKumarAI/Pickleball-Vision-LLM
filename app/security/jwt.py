"""Supabase JWT verification.

Supabase issues the JWT to the browser on login/signup; the API trusts it after
verifying the signature. Two modes:

* **HS256** with the project JWT secret (legacy / simplest — used in dev/tests).
* **RS256** via JWKS (Supabase signing keys) when ``supabase_jwks_url`` is set.

Returns the decoded claims; ``sub`` is the user id (= ``auth.uid()``).
"""

from __future__ import annotations

from typing import Any, Dict

import jwt
from jwt import PyJWKClient

from app.config import Settings

_jwks_client: PyJWKClient | None = None


class InvalidToken(Exception):
    """Raised when a token fails verification."""


def verify_supabase_jwt(token: str, settings: Settings) -> Dict[str, Any]:
    """Verify a Supabase JWT and return its claims, or raise :class:`InvalidToken`."""
    try:
        if settings.supabase_jwks_url:
            global _jwks_client
            if _jwks_client is None:
                _jwks_client = PyJWKClient(settings.supabase_jwks_url)
            signing_key = _jwks_client.get_signing_key_from_jwt(token).key
            return jwt.decode(
                token, signing_key, algorithms=["RS256", "ES256"],
                audience=settings.supabase_jwt_aud,
            )
        return jwt.decode(
            token, settings.supabase_jwt_secret, algorithms=["HS256"],
            audience=settings.supabase_jwt_aud,
        )
    except jwt.PyJWTError as e:
        raise InvalidToken(str(e)) from e


def claims_to_user(claims: Dict[str, Any]) -> Dict[str, Any]:
    """Map JWT claims to our user dict."""
    meta = claims.get("app_metadata", {}) or {}
    return {
        "id": claims.get("sub"),
        "email": claims.get("email"),
        "role": meta.get("role", claims.get("role", "authenticated")),
        "is_admin": meta.get("role") == "admin",
    }
