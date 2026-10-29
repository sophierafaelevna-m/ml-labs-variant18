"""Independent monitoring service for Lab 6."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import threading
import urllib.request
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/LAB6"
OUT.mkdir(parents=True, exist_ok=True)

INFERENCE_URL = os.getenv(
    "LAB6_INFERENCE_URL",
    "http://127.0.0.1:8000",
)

CSV_PATH = OUT / "monitoring_samples.csv"
LOG_PATH = OUT / "monitoring.log"
CHART_PATH = OUT / "monitoring_chart.png"

FIELDS = [
    "timestamp",
    "active_version",
    "predict_requests_total",
    "predict_successful",
    "predict_failed",
    "average_predict_latency_ms",
]

logger = logging.getLogger("lab6.monitoring")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)


class MonitorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_sample = None
        self.last_error = None
        self.sample_count = 0
        self.last_history_count = None

    def snapshot(self):
        with self.lock:
            return {
                "sample_count": self.sample_count,
                "last_sample": (dict(self.last_sample) if self.last_sample else None),
                "last_error": self.last_error,
            }


state = MonitorState()


def get_json(path: str) -> dict:
    with urllib.request.urlopen(
        INFERENCE_URL + path,
        timeout=3,
    ) as response:
        return json.load(response)


def collect_once():
    metrics = get_json("/metrics")
    registry = get_json("/models")

    timestamp = datetime.now(timezone.utc).isoformat()

    sample = {
        "timestamp": timestamp,
        "active_version": registry["active_version"],
        "predict_requests_total": metrics["predict_requests_total"],
        "predict_successful": metrics["predict_successful"],
        "predict_failed": metrics["predict_failed"],
        "average_predict_latency_ms": (metrics["average_predict_latency_ms"]),
    }

    with state.lock:
        if state.last_history_count is None:
            logger.info(
                "Monitoring started | active_version=%s",
                registry["active_version"],
            )
        else:
            new_events = registry["history"][state.last_history_count :]

            for event in new_events:
                logger.info(
                    "MODEL_EVENT | action=%s | version=%s | previous=%s | reason=%s",
                    event.get("action"),
                    event.get("version"),
                    event.get("previous_version"),
                    event.get("reason"),
                )

        state.last_history_count = len(registry["history"])

        if (
            state.last_sample is not None
            and state.last_sample["active_version"] != sample["active_version"]
        ):
            logger.info(
                "ACTIVE_MODEL_CHANGED | %s -> %s",
                state.last_sample["active_version"],
                sample["active_version"],
            )

        new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

        with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)

            if new_file:
                writer.writeheader()

            writer.writerow(sample)

        state.last_sample = sample
        state.last_error = None
        state.sample_count += 1


async def monitoring_loop():
    while True:
        try:
            await asyncio.to_thread(collect_once)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with state.lock:
                state.last_error = str(error)

            logger.error("MONITORING_ERROR | %s", error)

        await asyncio.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitoring_loop())

    try:
        yield
    finally:
        task.cancel()

        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Lab 6 — Independent Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    snapshot = state.snapshot()

    return {
        "status": (
            "ok"
            if snapshot["last_sample"] is not None and snapshot["last_error"] is None
            else "degraded"
        ),
        "inference_url": INFERENCE_URL,
    }


@app.get("/status")
def status():
    return state.snapshot()


@app.get("/history")
def history():
    try:
        registry = get_json("/models")
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    return {
        "active_version": registry["active_version"],
        "events": registry["history"],
    }


@app.get("/chart")
def chart():
    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No monitoring samples available",
        )

    with CSV_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Monitoring CSV is empty",
        )

    # Find the most recent period when v2 was active.
    v2_indices = [
        index for index, row in enumerate(rows) if row["active_version"] == "v2"
    ]

    if v2_indices:
        last_v2 = v2_indices[-1]
        first_v2 = last_v2

        while first_v2 > 0 and rows[first_v2 - 1]["active_version"] == "v2":
            first_v2 -= 1

        # Include samples before promotion and after rollback.
        start_index = max(0, first_v2 - 20)
        end_index = min(len(rows), last_v2 + 21)

        rows = rows[start_index:end_index]

    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]

    successful = [int(row["predict_successful"]) for row in rows]

    failed = [int(row["predict_failed"]) for row in rows]

    versions = [1 if row["active_version"] == "v1" else 2 for row in rows]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(
        times,
        successful,
        label="Successful requests",
        linewidth=2,
    )
    axes[0].plot(
        times,
        failed,
        label="Failed requests",
        linewidth=2,
    )

    axes[0].set_title(
        "Lab 6: Model switching under load",
        fontsize=15,
        pad=15,
    )
    axes[0].set_ylabel("Requests")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].step(
        times,
        versions,
        where="post",
        linewidth=2,
        label="Active model",
    )

    axes[1].set_yticks(
        [1, 2],
        labels=["v1", "v2"],
    )
    axes[1].set_ylim(0.8, 2.2)
    axes[1].set_ylabel("Model version")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        CHART_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return FileResponse(
        CHART_PATH,
        media_type="image/png",
    )
