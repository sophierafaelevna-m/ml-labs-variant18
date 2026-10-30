"""Reproducible HTTP benchmark for Lab 7 batch inference."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

API_URL = "http://127.0.0.1:8000/predict/batch"
OUTPUT = Path("reports/LAB7/parallel_benchmark.csv")

BATCH_SIZE = 32
WARMUP_RUNS = 5
REPEATS = 30

BASE_TEXTS = [
    "NASA launched a spacecraft to explore Mars.",
    "Scientists study planets and distant galaxies.",
    "The space telescope observed a new star.",
    "Computer graphics use rendering algorithms.",
    "Three dimensional images require graphics software.",
    "A new GPU improves image processing.",
    "The discussion concerns religious beliefs.",
    "People debate different theological ideas.",
    "The article discusses atheism and philosophy.",
    "Astronomers measured the orbit of a satellite.",
    "Digital image compression reduces file size.",
    "The debate focuses on faith and religion.",
]

TEXTS = [BASE_TEXTS[i % len(BASE_TEXTS)] for i in range(BATCH_SIZE)]


def request_batch(workers: int) -> dict:
    """Send one batch request and measure client-side latency."""
    payload = json.dumps({"texts": TEXTS, "workers": workers}).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urlopen(request, timeout=120) as response:
        data = json.loads(response.read())

    latency_ms = (time.perf_counter() - start) * 1000

    if data["count"] != BATCH_SIZE:
        raise RuntimeError("Incorrect number of predictions")

    if len(data["results"]) != BATCH_SIZE:
        raise RuntimeError("Incomplete prediction results")

    return {
        "latency_ms": latency_ms,
        "server_elapsed_ms": data["elapsed_ms"],
        "version": data["version"],
    }


def percentile_95(values: list[float]) -> float:
    """Calculate the nearest-rank 95th percentile."""
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[index]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cpu_count = max(1, os.cpu_count() or 1)
    worker_options = [workers for workers in (1, 2, 4) if workers <= cpu_count]

    rows = []
    baseline_mean_ms = None

    print(f"Logical CPU count: {cpu_count}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Measured repeats: {REPEATS}")
    print()

    for workers in worker_options:
        print(f"Testing workers={workers}")

        for _ in range(WARMUP_RUNS):
            request_batch(workers)

        measurements = []

        for repeat in range(1, REPEATS + 1):
            result = request_batch(workers)
            latency_ms = result["latency_ms"]
            throughput = BATCH_SIZE / (latency_ms / 1000)

            measurements.append(latency_ms)

            rows.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    "batch_size": BATCH_SIZE,
                    "warmup_runs": WARMUP_RUNS,
                    "latency_ms": round(latency_ms, 3),
                    "server_elapsed_ms": round(result["server_elapsed_ms"], 3),
                    "throughput_rps": round(throughput, 3),
                    "version": result["version"],
                }
            )

        mean_ms = statistics.mean(measurements)
        p95_ms = percentile_95(measurements)

        if workers == 1:
            baseline_mean_ms = mean_ms

        speedup = baseline_mean_ms / mean_ms

        print(f"  Mean latency: {mean_ms:.3f} ms")
        print(f"  P95 latency: {p95_ms:.3f} ms")
        print(f"  Mean throughput: {BATCH_SIZE / (mean_ms / 1000):.2f} texts/s")
        print(f"  Speedup: {speedup:.3f}x")
        print()

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "workers",
                "repeat",
                "batch_size",
                "warmup_runs",
                "latency_ms",
                "server_elapsed_ms",
                "throughput_rps",
                "version",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
