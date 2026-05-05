import os
import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("JWT_SECRET", "test-secret-key")


@pytest.fixture(scope="module")
def app():
    from fraudguard.api.main import app
    return app


@pytest.fixture(scope="module")
async def auth_token(app) -> str:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "secret"},
        )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def fraud_payload() -> dict:
    return {
        "msisdn": "+33687654321",
        "duration_sec": 3600,
        "destination": "+9XXXXXXXXX",
        "call_type": "INTERNATIONAL",
        "hour_of_day": 3,
        "calls_last_hour": 45,
        "sms_last_hour": 80,
        "unique_dest_24h": 50,
        "timestamp": "2024-01-15T03:15:00",
    }


@pytest.fixture
def legit_payload() -> dict:
    return {
        "msisdn": "+33612345678",
        "duration_sec": 180,
        "destination": "+33698765432",
        "call_type": "VOICE",
        "hour_of_day": 14,
        "calls_last_hour": 2,
        "sms_last_hour": 3,
        "unique_dest_24h": 5,
        "timestamp": "2024-01-15T14:30:00",
    }


@pytest.mark.anyio
async def test_predict_fraud_call(app, fraud_payload: dict, auth_token: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json=fraud_payload,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "FRAUD"
    assert data["risk_score"] >= 0.7


@pytest.mark.anyio
async def test_predict_legitimate_call(app, legit_payload: dict, auth_token: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json=legit_payload,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "LEGITIMATE"
    assert data["inference_latency_ms"] > 0


@pytest.mark.anyio
async def test_predict_returns_model_version(app, legit_payload: dict, auth_token: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            json=legit_payload,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1.0.0"


@pytest.mark.anyio
async def test_predict_requires_auth(app, legit_payload: dict) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/predict", json=legit_payload)
    assert response.status_code == 401


@pytest.mark.anyio
async def test_health_liveness(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_health_readiness(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.anyio
async def test_model_info(app, auth_token: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/model/info",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0"
    assert len(data["feature_names"]) == 6


@pytest.mark.anyio
async def test_drift_endpoint(app, auth_token: str) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/monitoring/drift",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert response.status_code == 200
    reports = response.json()
    assert len(reports) == 6
    for r in reports:
        assert "feature" in r
        assert "psi_score" in r
        assert "status" in r
