"""Phase 4/5 offline tests: Stripe webhook handling, account usage + GDPR delete,
admin gating."""

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.deps import get_repo
from app.main import create_app
from app.services.repo import InMemoryRepo
from app.services.stripe_service import handle_event

SECRET = "dev-insecure-change-me"


def _token(sub, role="user"):
    return jwt.encode({"sub": sub, "email": f"{sub}@x.com", "aud": "authenticated",
                       "app_metadata": {"role": role}}, SECRET, algorithm="HS256")


def _hdr(sub, role="user"):
    return {"Authorization": f"Bearer {_token(sub, role)}"}


@pytest.fixture()
def ctx():
    get_settings.cache_clear()
    repo = InMemoryRepo()
    app = create_app()
    app.dependency_overrides[get_repo] = lambda: repo
    return TestClient(app), repo


# --- Stripe webhook (pure handler) ---

def test_webhook_checkout_upgrades_plan():
    repo = InMemoryRepo()
    handle_event("checkout.session.completed",
                 {"metadata": {"user_id": "u1", "plan": "pro"}, "subscription": "sub_1"}, repo)
    assert repo.get_subscription("u1")["plan"] == "pro"


def test_webhook_subscription_deleted_downgrades():
    repo = InMemoryRepo()
    repo.set_subscription("u1", plan="pro", status="active")
    handle_event("customer.subscription.deleted", {"metadata": {"user_id": "u1"}}, repo)
    assert repo.get_subscription("u1")["plan"] == "free"


def test_webhook_ignores_eventless_metadata():
    repo = InMemoryRepo()
    assert handle_event("checkout.session.completed", {"metadata": {}}, repo) is None


# --- account ---

def test_account_usage(ctx):
    client, repo = ctx
    r = client.get("/account/usage", headers=_hdr("u1"))
    assert r.status_code == 200
    assert r.json()["plan"] == "free" and r.json()["videos_limit"] == 3


def test_gdpr_delete_purges(ctx):
    client, repo = ctx
    client.post("/jobs", json={"object_key": "u1/a.mp4"}, headers=_hdr("u1"))
    assert len(repo.list_jobs("u1")) == 1
    r = client.request("DELETE", "/account", headers=_hdr("u1"))
    assert r.status_code == 200 and r.json()["status"] == "erased"
    assert repo.list_jobs("u1") == []


# --- admin ---

def test_admin_requires_admin_claim(ctx):
    client, _ = ctx
    assert client.get("/admin/jobs", headers=_hdr("u1")).status_code == 403


def test_admin_lists_all_jobs(ctx):
    client, _ = ctx
    client.post("/jobs", json={"object_key": "u1/a.mp4"}, headers=_hdr("u1"))
    client.post("/jobs", json={"object_key": "u2/b.mp4"}, headers=_hdr("u2"))
    r = client.get("/admin/jobs", headers=_hdr("admin1", role="admin"))
    assert r.status_code == 200 and r.json()["count"] == 2
