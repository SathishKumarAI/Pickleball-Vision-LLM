"""Phase 2 offline tests: jobs control plane with in-memory repo + fake Modal.

Uses FastAPI dependency overrides so the whole flow (auth, quota, dedup,
ownership, cancel) runs with zero cloud/GPU.
"""

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.deps import get_dispatcher, get_repo, get_storage
from app.main import create_app
from app.services.dispatch import FakeDispatcher
from app.services.repo import InMemoryRepo
from app.services.storage import FakeStorage

SECRET = "dev-insecure-change-me"


def _token(sub, email="u@x.com", role="user"):
    return jwt.encode({"sub": sub, "email": email, "aud": "authenticated",
                       "app_metadata": {"role": role}}, SECRET, algorithm="HS256")


def _hdr(sub):
    return {"Authorization": f"Bearer {_token(sub)}"}


@pytest.fixture()
def ctx():
    get_settings.cache_clear()
    repo, disp, store = InMemoryRepo(), FakeDispatcher(), FakeStorage()
    app = create_app()
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_dispatcher] = lambda: disp
    app.dependency_overrides[get_storage] = lambda: store
    return TestClient(app), repo, disp


def test_upload_url_requires_auth(ctx):
    client, *_ = ctx
    assert client.post("/jobs/upload-url", json={"filename": "a.mp4"}).status_code == 401


def test_upload_url_minted(ctx):
    client, *_ = ctx
    r = client.post("/jobs/upload-url", json={"filename": "a.mp4"}, headers=_hdr("u1"))
    assert r.status_code == 200
    body = r.json()
    assert body["object_key"].startswith("u1/") and "upload=1" in body["upload_url"]


def test_create_job_spawns_modal(ctx):
    client, repo, disp = ctx
    r = client.post("/jobs", json={"object_key": "u1/x/a.mp4"}, headers=_hdr("u1"))
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert len(disp.spawns) == 1 and disp.spawns[0]["job_id"] == jid
    assert repo.get_job(jid)["modal_call_id"]


def test_dedup_by_sha_does_not_respawn(ctx):
    client, repo, disp = ctx
    payload = {"object_key": "u1/x/a.mp4", "content_sha256": "abc123"}
    r1 = client.post("/jobs", json=payload, headers=_hdr("u1"))
    r2 = client.post("/jobs", json=payload, headers=_hdr("u1"))
    assert r2.json()["deduplicated"] is True
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert len(disp.spawns) == 1  # second call did not spawn


def test_quota_blocks_over_limit(ctx):
    client, repo, _ = ctx
    # free plan = 3/mo; seed usage at the limit
    from app.routers.jobs import _period
    repo.incr_usage("u1", _period(), videos=3, seconds=0)
    r = client.post("/jobs", json={"object_key": "u1/x/a.mp4"}, headers=_hdr("u1"))
    assert r.status_code == 402
    assert r.json()["detail"]["error"] == "quota exceeded"


def test_ownership_enforced(ctx):
    client, *_ = ctx
    jid = client.post("/jobs", json={"object_key": "u1/x/a.mp4"}, headers=_hdr("u1")).json()["job_id"]
    assert client.get(f"/jobs/{jid}", headers=_hdr("u1")).status_code == 200
    assert client.get(f"/jobs/{jid}", headers=_hdr("u2")).status_code == 403
    assert client.get("/jobs/does-not-exist", headers=_hdr("u1")).status_code == 404


def test_list_jobs_owner_scoped(ctx):
    client, *_ = ctx
    client.post("/jobs", json={"object_key": "u1/a.mp4"}, headers=_hdr("u1"))
    client.post("/jobs", json={"object_key": "u1/b.mp4"}, headers=_hdr("u1"))
    client.post("/jobs", json={"object_key": "u2/c.mp4"}, headers=_hdr("u2"))
    assert client.get("/jobs", headers=_hdr("u1")).json()["count"] == 2
    assert client.get("/jobs", headers=_hdr("u2")).json()["count"] == 1


def test_cancel_flow(ctx):
    client, repo, disp = ctx
    jid = client.post("/jobs", json={"object_key": "u1/a.mp4"}, headers=_hdr("u1")).json()["job_id"]
    r = client.post(f"/jobs/{jid}/cancel", headers=_hdr("u1"))
    assert r.status_code == 200 and r.json()["status"] == "cancelling"
    assert repo.get_job(jid)["status"] == "cancelling"
    assert disp.cancelled  # hard-cancel attempted


def test_result_and_video_not_ready(ctx):
    client, *_ = ctx
    jid = client.post("/jobs", json={"object_key": "u1/a.mp4"}, headers=_hdr("u1")).json()["job_id"]
    assert client.get(f"/jobs/{jid}/result", headers=_hdr("u1")).status_code == 409
    assert client.get(f"/jobs/{jid}/video", headers=_hdr("u1")).status_code == 409


def test_quota_endpoint(ctx):
    client, *_ = ctx
    r = client.get("/jobs/quota", headers=_hdr("u1"))
    assert r.status_code == 200 and r.json()["plan"] == "free" and r.json()["limit"] == 3
