"""Window-based data drift monitoring for Lab 7."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.lab5.config import load_config
from src.lab5.generator import generate

OUTPUT = Path("reports/LAB7")
WINDOW_SIZE = 200
WINDOW_STEP = 100
STREAM_SIZE = 800
DRIFT_START = 400
PSI_THRESHOLD = 0.5

FEATURES = [
    "word_count",
    "char_count",
    "top_vocab_share",
    "source_class",
]


def make_samples(
    seed: int,
    count: int,
    scenario: str = "clean",
) -> pd.DataFrame:
    config = load_config("configs/lab5.yaml").model_copy(deep=True)
    config.seed = seed
    config.n_samples = count
    config.mode = "similar"

    if scenario == "length":
        config.drift.length_multiplier = 2.5

    elif scenario == "class_shift":
        config.drift.class_probabilities = {
            name: (0.85 if name == "sci.space" else 0.05) for name in config.classes
        }

    elif scenario == "vocabulary":
        config.mode = "random"

    elif scenario != "clean":
        raise ValueError(f"Unknown scenario: {scenario}")

    return generate(config)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{2,}", text.lower())


def build_features(
    data: pd.DataFrame,
    vocabulary: set[str],
) -> pd.DataFrame:
    records = []

    for text, source_class in zip(
        data["text"],
        data["target_name"],
        strict=True,
    ):
        words = tokenize(str(text))

        known = sum(word in vocabulary for word in words)

        records.append(
            {
                "word_count": len(str(text).split()),
                "char_count": len(str(text)),
                "top_vocab_share": (known / len(words) if words else 0.0),
                "source_class": str(source_class),
            }
        )

    return pd.DataFrame(records)


def numeric_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))

    if len(edges) == 1:
        value = edges[0]
        edges = np.array([value - 1e-9, value + 1e-9])

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    epsilon = 1e-4
    ref_share = ref_counts / len(ref) + epsilon
    cur_share = cur_counts / len(cur) + epsilon

    ref_share /= ref_share.sum()
    cur_share /= cur_share.sum()

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    categories = sorted(set(reference.astype(str)) | set(current.astype(str)))

    ref_share = (
        reference.astype(str)
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    cur_share = (
        current.astype(str)
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    epsilon = 1e-4
    ref_share = ref_share + epsilon
    cur_share = cur_share + epsilon

    ref_share /= ref_share.sum()
    cur_share /= cur_share.sum()

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def run_scenario(
    scenario: str,
    reference: pd.DataFrame,
    vocabulary: set[str],
) -> tuple[list[dict], dict]:
    before = make_samples(101, DRIFT_START)

    after = make_samples(
        102,
        STREAM_SIZE - DRIFT_START,
        scenario,
    )

    stream = pd.concat(
        [before, after],
        ignore_index=True,
    )

    features = build_features(stream, vocabulary)
    rows = []
    last_snapshot = None

    for start in range(
        0,
        STREAM_SIZE - WINDOW_SIZE + 1,
        WINDOW_STEP,
    ):
        end = start + WINDOW_SIZE
        current = features.iloc[start:end].reset_index(drop=True)

        scores = {
            "word_count_psi": numeric_psi(
                reference["word_count"],
                current["word_count"],
            ),
            "char_count_psi": numeric_psi(
                reference["char_count"],
                current["char_count"],
            ),
            "vocabulary_psi": numeric_psi(
                reference["top_vocab_share"],
                current["top_vocab_share"],
            ),
            "source_class_psi": categorical_psi(
                reference["source_class"],
                current["source_class"],
            ),
        }

        alert = any(score > PSI_THRESHOLD for score in scores.values())

        # Evidently evaluates every observation window.
        last_snapshot = Report(
            [
                DataDriftPreset(
                    columns=FEATURES,
                    method="psi",
                )
            ]
        ).run(current, reference)

        rows.append(
            {
                "scenario": scenario,
                "window_start": start,
                "window_end": end,
                **{key: round(value, 5) for key, value in scores.items()},
                "threshold": PSI_THRESHOLD,
                "alert": alert,
            }
        )

        print(
            f"  window {start:03d}-{end:03d}: "
            f"alert={alert}, "
            f"max_PSI={max(scores.values()):.3f}"
        )

    if last_snapshot is not None:
        last_snapshot.save_html(str(OUTPUT / f"evidently_{scenario}.html"))

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
        "drift_start": (DRIFT_START if scenario != "clean" else None),
        "first_detection_end": first_detection,
        "detection_delay_samples": delay,
        "false_alert_windows": false_alerts,
    }

    return rows, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    reference_data = make_samples(100, 400)

    word_counts = Counter(
        word for text in reference_data["text"] for word in tokenize(str(text))
    )
    vocabulary = set(word for word, _ in word_counts.most_common(100))

    reference = build_features(reference_data, vocabulary)

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
            vocabulary,
        )

        all_rows.extend(rows)
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(
        OUTPUT / "drift_metrics.csv",
        index=False,
    )

    with (OUTPUT / "drift_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\nDetection summary:")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nReports saved to:", OUTPUT)


if __name__ == "__main__":
    main()
