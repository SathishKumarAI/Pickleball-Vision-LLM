"""Phase 0 offline tests: FastAPI boot, health, JWT-gated auth, analyze path."""

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app

SECRET = "dev-insecure-change-me"  # matches Settings default


@pytest.fixture()
def client():
    get_settings.cache_clear()
    return TestClient(create_app())


def _token(sub="user-1", email="a@b.com", role="user", aud="authenticated"):
    return jwt.encode(
        {"sub": sub, "email": email, "aud": aud, "app_metadata": {"role": role}},
        SECRET, algorithm="HS256",
    )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_index_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "pickleball-vision-llm"


def test_auth_me_requires_token(client):
    assert client.get("/auth/me").status_code == 401


def test_auth_me_rejects_bad_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_auth_me_accepts_valid_token(client):
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {_token()}"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "user-1" and body["email"] == "a@b.com"


def test_admin_flag_from_claims(client):
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {_token(role='admin')}"})
    assert r.json()["is_admin"] is True


def test_analyze_requires_auth(client):
    assert client.post("/analyze", json={"detections": []}).status_code == 401


def test_analyze_runs_fusion_and_feedback(client):
    frames = [
        [{"bbox": [0, 0, 20, 40], "confidence": 0.9, "class_id": 0, "class_name": "person"},
         {"bbox": [100, 100, 110, 110], "confidence": 0.8, "class_id": 32, "class_name": "sports ball"}],
        [{"bbox": [0, 0, 20, 40], "confidence": 0.9, "class_id": 0, "class_name": "person"},
         {"bbox": [160, 160, 170, 170], "confidence": 0.8, "class_id": 32, "class_name": "sports ball"}],
        [{"bbox": [0, 0, 20, 40], "confidence": 0.9, "class_id": 0, "class_name": "person"}],
    ]
    r = client.post(
        "/analyze",
        json={"detections": frames, "fps": 30, "frame_height": 360, "backend": "rule"},
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["states"]) == 3
    assert [s["action"] for s in body["states"]] == ["serve-or-reset", "fast-exchange", "no-ball"]
    assert body["summary"]
