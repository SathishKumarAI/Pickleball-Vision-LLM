"""Minimal user database (SQLite, stdlib only).

Dependency-free persistence for the auth/login system. SQLite is fine for the
MVP and tests; swap for Postgres + SQLAlchemy when scaling (the ``UserDB``
interface is small and stable). Passwords are stored hashed (see
``src.services.auth``), never in plaintext.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "users.db"


class UserDB:
    """Thread-safe SQLite-backed user store."""

    def __init__(self, path: Optional[str] = None):
        self.path = str(path or _DEFAULT_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    email         TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name          TEXT,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_user(self, email: str, password_hash: str,
                    name: str = "") -> Dict[str, Any]:
        """Insert a user. Raises ValueError if the email already exists."""
        with self._lock, self._connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                    (email.lower().strip(), password_hash, name),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError("email already registered") from e
            return self._get(conn, "id = ?", (cur.lastrowid,))

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            return self._get(conn, "email = ?", (email.lower().strip(),))

    def get_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            return self._get(conn, "id = ?", (user_id,))

    @staticmethod
    def _get(conn: sqlite3.Connection, where: str, params: tuple) -> Optional[Dict[str, Any]]:
        row = conn.execute(f"SELECT * FROM users WHERE {where}", params).fetchone()
        return dict(row) if row else None


# Module-level singleton used by the auth blueprint.
db = UserDB()
