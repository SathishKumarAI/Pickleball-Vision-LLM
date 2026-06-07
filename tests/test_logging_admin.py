"""Offline tests: structured logging + admin log viewer + correlation id."""

import logging

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.deps import get_dispatcher, get_repo, get_storage
from app.logging_config import (
    ConsoleFormatter, JsonFormatter, RingBufferHandler, configure_logging,
    get_ring, request_id_var,
)
from app.main import create_app
from app.services.dispatch import FakeDispatcher
from app.services.repo import InMemoryRepo
from app.services.storage import FakeStorage

SECRET = "dev-insecure-change-me"


def _hdr(sub, role="user"):
    tok = jwt.encode({"sub": sub, "email": f"{sub}@x.com", "aud": "authenticated",
                      "app_metadata": {"role": role}}, SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture()
def client():
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_repo] = lambda: InMemoryRepo()
    app.dependency_overrides[get_dispatcher] = lambda: FakeDispatcher()
    app.dependency_overrides[get_storage] = lambda: FakeStorage()
    return TestClient(app)


def test_ring_buffer_captures_records():
    ring = RingBufferHandler(capacity=5)
    logging.getLogger("t").addHandler(ring)
    logging.getLogger("t").setLevel(logging.INFO)
    logging.getLogger("t").info("hello", extra={"k": "v"})
    recs = ring.records()
    assert recs[0]["msg"] == "hello" and recs[0]["k"] == "v"


def test_ring_buffer_level_filter_and_cap():
    ring = RingBufferHandler(capacity=3)
    lg = logging.getLogger("t2")
    lg.addHandler(ring); lg.setLevel(logging.DEBUG)
    for i in range(5):
        lg.info("m%d" % i)
    lg.error("boom")
    assert len(ring.buffer) == 3  # capped
    assert all(r["level"] == "ERROR" for r in ring.records(level="error"))


def test_console_and_json_formatters():
    rec = logging.makeLogRecord({"name": "x", "levelname": "INFO", "levelno": 20,
                                 "msg": "hi", "args": ()})
    out = ConsoleFormatter(color=False).format(rec)
    assert "ok:" in out and "[x]" in out and "hi" in out
    import json
    j = json.loads(JsonFormatter().format(rec))
    assert j["level"] == "INFO" and j["msg"] == "hi"


def test_correlation_id_header(client):
    r = client.get("/health")
    assert "X-Request-ID" in r.headers and "X-Response-Time-ms" in r.headers


def test_admin_logs_requires_admin(client):
    assert client.get("/admin/logs", headers=_hdr("u1")).status_code == 403


def test_admin_logs_returns_records(client):
    configure_logging()
    # generate a request so the middleware logs into the ring
    client.get("/health")
    r = client.get("/admin/logs", headers=_hdr("admin1", role="admin"))
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert all("level" in rec and "msg" in rec for rec in body["logs"])
