"""Explore joint rejection thresholds using development data only."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.length_guard import token_lengths
from src.lab4.model_confidence import (
    load_frozen_model,
    predict_with_confidence,
)
from src.lab4.validator import ValidatorStats, distance_to_training

FOREIGN_CATEGORIES = (
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "misc.forsale",
)


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()
    output_dir = Path("reports/LAB4")

    print("1. Reading existing TF-IDF diagnostic scores...")
    with (output_dir / "topic_similarity_diagnostic.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    id_topic = np.asarray(
        [float(row["similarity"]) for row in rows if row["group"] == "id_validation"]
    )
    foreign_topic = np.asarray(
        [float(row["similarity"]) for row in rows if row["group"] != "id_validation"]
    )

    assert len(id_topic) == 410
    assert len(foreign_topic) == 200

    print("2. Loading development texts...")
    val_records = list(iter_split_records(lab3, "val"))
    val_texts = [row[lab3.data.text_column] for row in val_records]

    foreign_texts = []
    for category in FOREIGN_CATEGORIES:
        dataset = fetch_20newsgroups(
            subset="test",
            categories=[category],
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        )
        selected = [
            text for text in dataset.data if text.strip() and len(text.split()) >= 5
        ][:50]
        assert len(selected) == 50
        foreign_texts.extend(selected)

    assert len(val_texts) == len(id_topic)
    assert len(foreign_texts) == len(foreign_topic)

    print("3. Loading frozen model and training statistics...")
    model, vocabulary = load_frozen_model(lab4.source.checkpoint_path)

    with np.load(
        output_dir / "validator_params.npz",
        allow_pickle=False,
    ) as params:
        stats = ValidatorStats(
            class_ids=params["class_ids"].copy(),
            counts=params["counts"].copy(),
            means=params["means"].copy(),
            variances=params["variances"].copy(),
        )
        epsilon = float(params["epsilon"])
        length_q99 = float(params["length_limit"])

    all_texts = val_texts + foreign_texts
    features = extract_feature_matrix(
        all_texts,
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )
    distances = distance_to_training(features, stats, epsilon=epsilon)
    _, confidences = predict_with_confidence(all_texts, model, vocabulary)
    lengths = token_lengths(all_texts)

    n_id = len(val_texts)
    id_distances = distances[:n_id]
    foreign_distances = distances[n_id:]
    id_confidences = confidences[:n_id]
    foreign_confidences = confidences[n_id:]
    id_lengths = lengths[:n_id]
    foreign_lengths = lengths[n_id:]

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(n_id)
    calibration_indices = shuffled[: n_id // 2]

    print("4. Exploring joint thresholds...")
    topic_quantiles = [0.02, 0.03, 0.05, 0.07, 0.08, 0.10]
    confidence_thresholds = [0.0, 0.25, 0.35, 0.40, 0.45, 0.48]
    distance_thresholds = [3.0, 4.0, 5.0, 1e9]
    length_thresholds = [length_q99, 3761.99, 1e9]

    candidates = []

    for topic_q, confidence_t, distance_t, length_t in itertools.product(
        topic_quantiles,
        confidence_thresholds,
        distance_thresholds,
        length_thresholds,
    ):
        topic_t = float(np.quantile(id_topic[calibration_indices], topic_q))

        id_rejected = (
            (id_topic < topic_t)
            | (id_confidences < confidence_t)
            | (id_distances > distance_t)
            | (id_lengths > length_t)
        )
        foreign_rejected = (
            (foreign_topic < topic_t)
            | (foreign_confidences < confidence_t)
            | (foreign_distances > distance_t)
            | (foreign_lengths > length_t)
        )

        calibration_frr = float(id_rejected[calibration_indices].mean())
        foreign_recall = float(foreign_rejected.mean())

        candidates.append(
            {
                "topic_quantile": topic_q,
                "topic_threshold": topic_t,
                "confidence_threshold": confidence_t,
                "distance_threshold": distance_t,
                "length_threshold": length_t,
                "calibration_false_reject_rate": calibration_frr,
                "development_foreign_recall": foreign_recall,
                "calibration_rejected": int(id_rejected[calibration_indices].sum()),
                "foreign_rejected": int(foreign_rejected.sum()),
            }
        )

    target_frr = 0.08
    target_recall = lab4.decision.min_foreign_recall

    eligible = [
        row for row in candidates if row["calibration_false_reject_rate"] <= target_frr
    ]
    eligible.sort(
        key=lambda row: (
            -row["development_foreign_recall"],
            row["calibration_false_reject_rate"],
        )
    )

    output_path = output_dir / "joint_calibration_candidates.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(eligible)

    print()
    print("Total configurations:", len(candidates))
    print("Within 8% calibration FRR:", len(eligible))
    print(
        "Meeting both development targets:",
        sum(row["development_foreign_recall"] >= target_recall for row in eligible),
    )

    print()
    print("TOP 10 CONFIGURATIONS")
    print("Topic | Confidence | Distance | Length | Calibration FRR | Foreign recall")
    print("-" * 85)

    for row in eligible[:10]:
        print(
            f"{row['topic_threshold']:.5f} | "
            f"{row['confidence_threshold']:.2f} | "
            f"{row['distance_threshold']:.1f} | "
            f"{row['length_threshold']:.2f} | "
            f"{row['calibration_false_reject_rate']:.2%} | "
            f"{row['development_foreign_recall']:.2%}"
        )

    print()
    print("Saved:", output_path)
    print("PASS: joint threshold exploration completed")


if __name__ == "__main__":
    main()
