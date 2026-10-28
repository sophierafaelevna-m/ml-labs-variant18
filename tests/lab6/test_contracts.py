"""Contract and registry tests for Lab 6 FastAPI service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.lab6 import service
from src.lab6.registry import REGISTRY_PATH, ModelRegistry


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """Create an isolated API instance with a temporary registry."""

    registry_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    # Start every test from the same known state.
    registry_data["active_version"] = "v1"
    registry_data["rollback_target"] = None
    registry_data["versions"]["v1"]["status"] = "working"
    registry_data["versions"]["v2"]["status"] = "canary"

    temporary_registry = tmp_path / "registry.json"
    temporary_registry.write_text(
        json.dumps(registry_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service,
        "ModelRegistry",
        lambda: ModelRegistry(temporary_registry),
    )

    with TestClient(service.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["active_version"] == "v1"


def test_predict_valid(client):
    response = client.post(
        "/predict",
        json={"text": "NASA launched a satellite into space."},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["version"] == "v1"
    assert 0 <= result["predicted_class"] < 4
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": 123},
        {"text": ""},
        {"text": "   "},
        {"text": "a" * 20_001},
        {"text": "Hello", "unknown_field": 42},
    ],
)
def test_predict_invalid_input(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_models(client):
    response = client.get("/models")

    assert response.status_code == 200

    result = response.json()

    assert result["active_version"] == "v1"
    assert set(result["versions"]) == {"v1", "v2"}

    for model in result["versions"].values():
        assert len(model["sha256"]) == 64
        assert model["metrics"]


def test_metrics(client):
    response = client.post(
        "/predict",
        json={"text": "The spacecraft entered orbit."},
    )
    assert response.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200

    metrics = response.json()

    assert metrics["predict_requests_total"] == 1
    assert metrics["predict_successful"] == 1
    assert metrics["predict_failed"] == 0


def test_promote_and_rollback(client):
    promote = client.post(
        "/promote",
        json={
            "version": "v2",
            "reason": "Automated contract test",
        },
    )

    assert promote.status_code == 200
    assert promote.json()["active_version"] == "v2"

    prediction = client.post(
        "/predict",
        json={"text": "NASA spacecraft mission."},
    )

    assert prediction.status_code == 200
    assert prediction.json()["version"] == "v2"

    rollback = client.post("/rollback")

    assert rollback.status_code == 200
    assert rollback.json()["active_version"] == "v1"

    health = client.get("/health")

    assert health.json()["active_version"] == "v1"

    registry = client.get("/models").json()

    assert registry["versions"]["v1"]["status"] == "working"
    assert registry["versions"]["v2"]["status"] == "canary"
    assert registry["history"][-2]["action"] == "promote"
    assert registry["history"][-1]["action"] == "rollback"


def test_promote_unknown_version(client):
    response = client.post(
        "/promote",
        json={
            "version": "v999",
            "reason": "Negative control",
        },
    )

    assert response.status_code == 404


def test_promote_active_version(client):
    response = client.post(
        "/promote",
        json={
            "version": "v1",
            "reason": "Negative control",
        },
    )

    assert response.status_code == 409


def test_rollback_without_promotion(client):
    response = client.post("/rollback")

    assert response.status_code == 409


def test_promote_invalid_request(client):
    response = client.post(
        "/promote",
        json={"version": "v2"},
    )

    assert response.status_code == 422


def test_reject_wrong_model_hash(tmp_path):
    """A modified expected checksum must prevent model loading."""

    registry_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    registry_data["versions"]["v1"]["sha256"] = "0" * 64

    temporary_registry = tmp_path / "bad_registry.json"
    temporary_registry.write_text(
        json.dumps(registry_data, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ModelRegistry(temporary_registry)


def test_100_requests_to_each_endpoint(client):
    """Send 100 successful requests to each of the six API routes."""

    for _ in range(100):
        assert client.get("/health").status_code == 200

        assert (
            client.post(
                "/predict",
                json={"text": "NASA spacecraft mission."},
            ).status_code
            == 200
        )

        assert client.get("/models").status_code == 200

        assert client.get("/metrics").status_code == 200

        assert (
            client.post(
                "/promote",
                json={
                    "version": "v2",
                    "reason": "Repeated switching test",
                },
            ).status_code
            == 200
        )

        assert client.post("/rollback").status_code == 200

    metrics = client.get("/metrics").json()

    assert metrics["predict_requests_total"] == 100
    assert metrics["predict_successful"] == 100
    assert metrics["predict_failed"] == 0
    assert client.get("/health").json()["active_version"] == "v1"


def test_http_status_metrics(client):
    # Successful request
    response = client.get("/health")
    assert response.status_code == 200

    # Invalid input must return 422
    response = client.post(
        "/predict",
        json={"text": 123},
    )
    assert response.status_code == 422

    # Unknown endpoint must return 404
    response = client.get("/unknown-endpoint")
    assert response.status_code == 404

    # Check accumulated HTTP metrics
    response = client.get("/metrics")
    assert response.status_code == 200

    metrics = response.json()

    assert metrics["http_requests_total"] == 3
    assert metrics["http_status_counts"]["200"] == 1
    assert metrics["http_status_counts"]["422"] == 1
    assert metrics["http_status_counts"]["404"] == 1

    assert metrics["http_validation_errors_422"] == 1
    assert metrics["http_server_errors_5xx"] == 0

    # Invalid input never reached model inference.
    assert metrics["predict_requests_total"] == 0
