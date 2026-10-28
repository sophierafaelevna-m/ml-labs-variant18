"""Concurrent HTTP load test with promotion and rollback."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
TOTAL_REQUESTS = 1000
WORKERS = 8
REPORT_PATH = Path("reports/LAB6/load_test.json")


def request_json(
    path: str,
    payload: dict | None = None,
) -> tuple[int, dict]:
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode("utf-8")}


def predict_once(number: int) -> dict:
    start = time.perf_counter()

    try:
        status, response = request_json(
            "/predict",
            {"text": f"NASA spacecraft satellite mission number {number}."},
        )

        success = (
            status == 200
            and response.get("version") in {"v1", "v2"}
            and isinstance(response.get("predicted_class"), int)
        )

        return {
            "request_id": number,
            "status": status,
            "success": success,
            "version": response.get("version"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        }

    except Exception as error:
        return {
            "request_id": number,
            "status": None,
            "success": False,
            "version": None,
            "error": repr(error),
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        }


def post_action(path: str, payload: dict) -> dict:
    status, response = request_json(path, payload)

    if status != 200:
        raise RuntimeError(f"{path} failed: HTTP {status}, {response}")

    print(f"{path}: {response}")
    return response


def main() -> None:
    status, health = request_json("/health")

    if status != 200 or health.get("active_version") != "v1":
        raise RuntimeError(
            f"Start the load test with v1 active: HTTP {status}, {health}"
        )

    results = []
    events = []
    promoted = False
    rolled_back = False

    print(f"Starting {TOTAL_REQUESTS} requests, {WORKERS} workers")
    start = time.perf_counter()

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [
                executor.submit(predict_once, number)
                for number in range(TOTAL_REQUESTS)
            ]

            for future in as_completed(futures):
                results.append(future.result())
                completed = len(results)

                if completed == 100 and not promoted:
                    response = post_action(
                        "/promote",
                        {
                            "version": "v2",
                            "reason": "Concurrent load test",
                        },
                    )
                    events.append(
                        {
                            "completed_requests": completed,
                            "action": "promote",
                            "response": response,
                        }
                    )
                    promoted = True

                if completed == 500 and not rolled_back:
                    response = post_action("/rollback", {})
                    events.append(
                        {
                            "completed_requests": completed,
                            "action": "rollback",
                            "response": response,
                        }
                    )
                    rolled_back = True

    finally:
        # Restore v1 if promotion succeeded but rollback failed.
        _, current_health = request_json("/health")

        if current_health.get("active_version") == "v2":
            post_action("/rollback", {})

    duration = time.perf_counter() - start

    successful = sum(result["success"] for result in results)
    failed = len(results) - successful
    lost = TOTAL_REQUESTS - len(results)

    success_rate = successful / TOTAL_REQUESTS * 100

    versions = dict(
        Counter(result["version"] for result in results if result["success"])
    )

    latencies = sorted(result["latency_ms"] for result in results if result["success"])

    p95 = (
        latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        if latencies
        else None
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "total_requests": TOTAL_REQUESTS,
        "completed_requests": len(results),
        "successful_requests": successful,
        "failed_requests": failed,
        "lost_requests": lost,
        "success_rate_percent": round(success_rate, 3),
        "duration_seconds": round(duration, 3),
        "throughput_requests_per_second": round(TOTAL_REQUESTS / duration, 3),
        "p95_latency_ms": p95,
        "versions_observed": versions,
        "switching_events": events,
        "errors": [result for result in results if not result["success"]][:20],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== LOAD TEST RESULTS ===")
    print("Requests sent:", TOTAL_REQUESTS)
    print("Responses received:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Lost:", lost)
    print(f"Success rate: {success_rate:.3f}%")
    print("Versions observed:", versions)
    print("P95 latency (ms):", p95)
    print("Report:", REPORT_PATH)

    if (
        success_rate < 99.9
        or failed != 0
        or lost != 0
        or not {"v1", "v2"}.issubset(versions)
        or len(events) != 2
    ):
        raise AssertionError("Load test acceptance criteria not met")


if __name__ == "__main__":
    main()
