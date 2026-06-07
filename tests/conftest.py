"""Shared pytest fixtures.

Repoints the singleton UserDB at a throwaway SQLite file so tests never touch
real data and stay isolated. Re-initialising the existing instance (rather than
replacing it) keeps every imported ``db`` reference valid.
"""

import os
import tempfile

import pytest


@pytest.fixture()
def app():
    from src.services.db import db
    tmp = os.path.join(tempfile.mkdtemp(), "users_test.db")
    db.__init__(tmp)  # repoint singleton at a fresh DB + create schema
    from src.api import create_app
    application = create_app()
    application.config.update(TESTING=True, SECRET_KEY="test-secret")
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth(client):
    """Register a user; return (headers, user_dict)."""
    r = client.post("/auth/register",
                    json={"email": "p@q.com", "password": "hunter2pw", "name": "P"})
    token = r.get_json()["token"]
    return {"Authorization": f"Bearer {token}"}, r.get_json()["user"]
