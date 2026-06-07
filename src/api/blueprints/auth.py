"""Auth API — register / login / current-user, plus a token-required guard.

Stateless bearer-token auth (``Authorization: Bearer <token>``) backed by the
SQLite ``UserDB`` and signed tokens from ``src.services.auth``. Importing this
module pulls only Flask + stdlib, so it's fully testable offline.
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, current_app, g, jsonify, request

from src.services.auth import hash_password, issue_token, verify_password, verify_token
from src.services.db import db

bp = Blueprint("auth", __name__)


def token_required(view):
    """Guard a route: require a valid bearer token; sets ``g.user``."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else None
        payload = verify_token(token, current_app.config.get("SECRET_KEY")) if token else None
        if not payload:
            return jsonify(error="authentication required"), 401
        user = db.get_by_id(payload["uid"])
        if not user:
            return jsonify(error="user no longer exists"), 401
        g.user = user
        return view(*args, **kwargs)
    return wrapper


@bp.post("/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify(error="email and password required"), 400
    if len(password) < 8:
        return jsonify(error="password must be at least 8 characters"), 400
    try:
        user = db.create_user(email, hash_password(password), data.get("name", ""))
    except ValueError as e:
        return jsonify(error=str(e)), 409
    token = issue_token(user["id"], user["email"], current_app.config.get("SECRET_KEY"))
    return jsonify(token=token, user=_public(user)), 201


@bp.post("/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    user = db.get_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify(error="invalid credentials"), 401
    token = issue_token(user["id"], user["email"], current_app.config.get("SECRET_KEY"))
    return jsonify(token=token, user=_public(user))


@bp.get("/auth/me")
@token_required
def me():
    return jsonify(user=_public(g.user))


def _public(user: dict) -> dict:
    """Strip secrets before returning a user."""
    return {"id": user["id"], "email": user["email"], "name": user.get("name", "")}
