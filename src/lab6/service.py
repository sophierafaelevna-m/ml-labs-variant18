"""FastAPI inference service for Lab 6."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.lab6.registry import ModelRegistry
from src.lab6.schemas import (
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    SwitchResponse,
)


class RequestMetrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.total_latency_ms = 0.0

    def record(self, success: bool, latency_ms: float) -> None:
        with self.lock:
            self.total += 1

            if success:
                self.successful += 1
            else:
                self.failed += 1

            self.total_latency_ms += latency_ms

    def snapshot(self) -> dict:
        with self.lock:
            average = self.total_latency_ms / self.total if self.total else 0.0

            return {
                "predict_requests_total": self.total,
                "predict_successful": self.successful,
                "predict_failed": self.failed,
                "average_predict_latency_ms": round(average, 3),
            }


class HTTPMetrics:
    """Thread-safe metrics for all HTTP responses."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total = 0
        self.status_counts: dict[str, int] = {}
        self.total_latency_ms = 0.0

    def record(self, status_code: int, latency_ms: float) -> None:
        with self.lock:
            self.total += 1

            key = str(status_code)
            self.status_counts[key] = self.status_counts.get(key, 0) + 1

            self.total_latency_ms += latency_ms

    def snapshot(self) -> dict:
        with self.lock:
            counts = dict(self.status_counts)

            average = self.total_latency_ms / self.total if self.total else 0.0

            return {
                "http_requests_total": self.total,
                "http_status_counts": counts,
                "http_validation_errors_422": counts.get("422", 0),
                "http_server_errors_5xx": sum(
                    count
                    for status, count in counts.items()
                    if 500 <= int(status) <= 599
                ),
                "http_average_latency_ms": round(average, 3),
            }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    app.state.metrics = RequestMetrics()
    app.state.http_metrics = HTTPMetrics()
    yield


app = FastAPI(
    title="Lab 6 — Model Serving",
    description="20 Newsgroups classification with model switching",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        request.app.state.http_metrics.record(500, elapsed_ms)
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000

    request.app.state.http_metrics.record(
        response.status_code,
        elapsed_ms,
    )

    return response


@app.get("/health")
def health(request: Request):
    registry = request.app.state.registry

    return {
        "status": "ok",
        "active_version": registry.get_active_model().version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    registry = request.app.state.registry
    metrics = request.app.state.metrics

    # Capture one model reference for the entire request.
    model = registry.get_active_model()

    start = time.perf_counter()
    success = False

    try:
        result = model.predict(payload.text)
        success = True
        return result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.record(success, elapsed_ms)


@app.get("/models")
def models(request: Request):
    return request.app.state.registry.get_registry()


@app.get("/metrics")
def metrics(request: Request):
    result = request.app.state.metrics.snapshot()

    result.update(request.app.state.http_metrics.snapshot())

    result["active_version"] = request.app.state.registry.get_active_model().version

    return result


@app.post("/promote", response_model=SwitchResponse)
def promote(payload: PromoteRequest, request: Request):
    registry = request.app.state.registry

    try:
        return registry.promote(
            version=payload.version,
            reason=payload.reason,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error


@app.post("/rollback", response_model=SwitchResponse)
def rollback(request: Request):
    registry = request.app.state.registry

    try:
        return registry.rollback()
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error
