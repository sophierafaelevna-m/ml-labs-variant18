"""Parallel batch inference for Lab 7."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.lab6.schemas import PredictRequest, PredictResponse

MAX_WORKERS = max(1, os.cpu_count() or 1)

router = APIRouter()


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[StrictStr] = Field(min_length=1, max_length=64)
    workers: int = Field(default=1, ge=1, le=MAX_WORKERS)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        for text in texts:
            PredictRequest(text=text)
        return texts


class BatchPredictResponse(BaseModel):
    version: str
    workers: int
    count: int
    elapsed_ms: float
    results: list[PredictResponse]


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request):
    registry = request.app.state.registry
    metrics = request.app.state.metrics

    # One model snapshot for the whole batch.
    model = registry.get_active_model()

    def predict_one(text: str) -> dict:
        start = time.perf_counter()
        success = False

        try:
            result = model.predict(text)
            success = True
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics.record(success, elapsed_ms)

    batch_start = time.perf_counter()

    if payload.workers == 1:
        results = [predict_one(text) for text in payload.texts]
    else:
        results = Parallel(
            n_jobs=payload.workers,
            prefer="threads",
        )(delayed(predict_one)(text) for text in payload.texts)

    elapsed_ms = (time.perf_counter() - batch_start) * 1000

    return BatchPredictResponse(
        version=model.version,
        workers=payload.workers,
        count=len(results),
        elapsed_ms=round(elapsed_ms, 3),
        results=results,
    )
