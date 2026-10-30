"""Monitor drift in predictions of the running FastAPI service."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.lab7.drift_monitor import (
    DRIFT_START,
    OUTPUT,
    PSI_THRESHOLD,
    STREAM_SIZE,
    WINDOW_SIZE,
    WINDOW_STEP,
    categorical_psi,
    make_samples,
    numeric_psi,
)

API_URL = "http://127.0.0.1:8000/predict/batch"
BATCH_SIZE = 32
PREDICTION_COLUMNS = ["predicted_class", "confidence"]


def predict_texts(texts: list[str]) -> tuple[pd.DataFrame, str]:
    """Obtain real predictions from the Lab 7 API."""
    records = []
    versions = set()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = [str(text)[:20_000] for text in texts[start : start + BATCH_SIZE]]

        payload = json.dumps(
            {
                "texts": batch,
                "workers": 1,
            }
        ).encode("utf-8")

        request = Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())

        predictions = result["results"]

        if len(predictions) != len(batch):
            raise RuntimeError("Incomplete batch response")

        versions.add(result["version"])

        for prediction in predictions:
            versions.add(prediction["version"])
            records.append(
                {
                    "predicted_class": str(prediction["predicted_class"]),
                    "confidence": float(prediction["confidence"]),
                }
            )

    if len(versions) != 1:
        raise RuntimeError(f"Model version changed during monitoring: {versions}")

    return pd.DataFrame(records), versions.pop()


def run_scenario(
    scenario: str,
    reference: pd.DataFrame,
    before: pd.DataFrame,
    version: str,
) -> tuple[list[dict], dict]:
    """Evaluate prediction drift in sliding windows."""
    after_data = make_samples(
        seed=102,
        count=STREAM_SIZE - DRIFT_START,
        scenario=scenario,
    )

    after, after_version = predict_texts(after_data["text"].tolist())

    if after_version != version:
        raise RuntimeError("Model version changed")

    stream = pd.concat(
        [before, after],
        ignore_index=True,
    )

    rows = []
    selected_snapshot = None
    last_snapshot = None

    for start in range(
        0,
        STREAM_SIZE - WINDOW_SIZE + 1,
        WINDOW_STEP,
    ):
        end = start + WINDOW_SIZE

        current = stream.iloc[start:end].reset_index(drop=True)

        class_psi = categorical_psi(
            reference["predicted_class"],
            current["predicted_class"],
        )

        confidence_psi = numeric_psi(
            reference["confidence"],
            current["confidence"],
        )

        alert = class_psi > PSI_THRESHOLD or confidence_psi > PSI_THRESHOLD

        snapshot = Report(
            [
                DataDriftPreset(
                    columns=PREDICTION_COLUMNS,
                    method="psi",
                )
            ]
        ).run(current, reference)

        last_snapshot = snapshot

        if alert and selected_snapshot is None:
            selected_snapshot = snapshot

        rows.append(
            {
                "scenario": scenario,
                "window_start": start,
                "window_end": end,
                "model_version": version,
                "predicted_class_psi": round(class_psi, 5),
                "confidence_psi": round(confidence_psi, 5),
                "threshold": PSI_THRESHOLD,
                "alert": alert,
            }
        )

        print(
            f"  window {start:03d}-{end:03d}: "
            f"class_PSI={class_psi:.3f}, "
            f"confidence_PSI={confidence_psi:.3f}, "
            f"alert={alert}"
        )

    # Save the first alert window, or the final clean window.
    report = selected_snapshot or last_snapshot

    if report is not None:
        report.save_html(str(OUTPUT / f"predictions_{scenario}.html"))

    alerts = [row for row in rows if row["alert"]]

    if scenario == "clean":
        first_detection = None
        delay = None
        false_alerts = len(alerts)
    else:
        detected = [row for row in alerts if row["window_end"] > DRIFT_START]

        first_detection = detected[0]["window_end"] if detected else None

        delay = first_detection - DRIFT_START if first_detection is not None else None

        false_alerts = sum(
            row["alert"] for row in rows if row["window_end"] <= DRIFT_START
        )

    summary = {
        "scenario": scenario,
        "model_version": version,
        "first_detection_end": first_detection,
        "detection_delay_samples": delay,
        "false_alert_windows": false_alerts,
    }

    return rows, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Obtaining reference predictions...")

    reference_data = make_samples(
        seed=100,
        count=400,
    )

    reference, version = predict_texts(reference_data["text"].tolist())

    print(f"Model version: {version}")
    print("Obtaining baseline stream predictions...")

    before_data = make_samples(
        seed=101,
        count=DRIFT_START,
    )

    before, before_version = predict_texts(before_data["text"].tolist())

    if before_version != version:
        raise RuntimeError("Model version changed")

    all_rows = []
    summaries = []

    for scenario in (
        "clean",
        "length",
        "class_shift",
        "vocabulary",
    ):
        print(f"\nScenario: {scenario}")

        rows, summary = run_scenario(
            scenario,
            reference,
            before,
            version,
        )

        all_rows.extend(rows)
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(
        OUTPUT / "prediction_drift_metrics.csv",
        index=False,
    )

    pd.DataFrame(summaries).to_csv(
        OUTPUT / "prediction_drift_summary.csv",
        index=False,
    )

    print("\nPrediction drift summary:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nReports saved to:", OUTPUT)


if __name__ == "__main__":
    main()
