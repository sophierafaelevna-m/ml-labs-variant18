"""Tests for Lab 7 parallel batch inference."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.lab7.batch import MAX_WORKERS, router


class FakeModel:
    version = "v1"

    def predict(self, text: str) -> dict:
        if text == "SIMULATE_FAILURE":
            raise RuntimeError("Simulated worker failure")

        return {
            "version": self.version,
            "predicted_class": 3,
            "confidence": 0.9,
        }


class FakeRegistry:
    def __init__(self):
        self.model = FakeModel()

    def get_active_model(self):
        return self.model


class FakeMetrics:
    def __init__(self):
        self.records = []

    def record(self, success: bool, elapsed_ms: float):
        self.records.append((success, elapsed_ms))


def make_client() -> TestClient:
    app = FastAPI()
    app.state.registry = FakeRegistry()
    app.state.metrics = FakeMetrics()
    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(
        app,
        raise_server_exceptions=False,
    )


def test_sequential_batch():
    client = make_client()

    response = client.post(
        "/predict/batch",
        json={
            "texts": ["NASA", "Space", "Graphics"],
            "workers": 1,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 3
    assert data["workers"] == 1
    assert data["version"] == "v1"
    assert len(data["results"]) == 3


def test_parallel_batch():
    client = make_client()
    workers = min(2, MAX_WORKERS)

    response = client.post(
        "/predict/batch",
        json={
            "texts": ["NASA", "Space", "Graphics"],
            "workers": workers,
        },
    )

    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_empty_text_rejected():
    client = make_client()

    response = client.post(
        "/predict/batch",
        json={
            "texts": [""],
            "workers": 1,
        },
    )

    assert response.status_code == 422


def test_workers_limit():
    client = make_client()

    response = client.post(
        "/predict/batch",
        json={
            "texts": ["NASA"],
            "workers": MAX_WORKERS + 1,
        },
    )

    assert response.status_code == 422


def test_worker_error_does_not_kill_service():
    client = make_client()
    workers = min(2, MAX_WORKERS)

    failed_response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "NASA",
                "SIMULATE_FAILURE",
                "Graphics",
            ],
            "workers": workers,
        },
    )

    # A task failure may fail this request.
    assert failed_response.status_code == 500

    # But the API process must remain available.
    health_response = client.get("/health")
    assert health_response.status_code == 200

    # A subsequent batch must still work.
    recovery_response = client.post(
        "/predict/batch",
        json={
            "texts": ["NASA", "Graphics"],
            "workers": workers,
        },
    )

    assert recovery_response.status_code == 200
    assert recovery_response.json()["count"] == 2
