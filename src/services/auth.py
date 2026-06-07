"""Authentication helpers — password hashing + signed bearer tokens.

Uses only libraries that ship with Flask:

* ``werkzeug.security`` — salted password hashing (pbkdf2).
* ``itsdangerous`` — signed, expiring tokens (stateless bearer auth, no PyJWT
  dependency). Swap for JWT later if cross-service verification is needed.

Token secret comes from ``APP_SECRET`` / Flask ``SECRET_KEY``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

_SALT = "pvllm-auth"
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def _serializer(secret: Optional[str] = None) -> URLSafeTimedSerializer:
    secret = secret or os.getenv("APP_SECRET", "dev-insecure-change-me")
    return URLSafeTimedSerializer(secret, salt=_SALT)


def issue_token(user_id: int, email: str, secret: Optional[str] = None) -> str:
    """Create a signed bearer token for a user."""
    return _serializer(secret).dumps({"uid": user_id, "email": email})


def verify_token(token: str, secret: Optional[str] = None,
                 max_age: int = TOKEN_TTL_SECONDS) -> Optional[Dict[str, Any]]:
    """Return the token payload, or None if invalid/expired."""
    try:
        return _serializer(secret).loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
