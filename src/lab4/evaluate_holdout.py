"""Evaluate the frozen Lab 4 validator on held-out ID test data."""

from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import numpy as np

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.length_guard import reject_with_length
from src.lab4.model_confidence import (
    load_frozen_model,
    predict_with_confidence,
)
from src.lab4.validator import (
    ValidatorStats,
    distance_to_training,
    reject_predictions,
)


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()

    output_dir = Path("reports/LAB4")
    params_path = output_dir / "validator_params.npz"

    print("1. Loading frozen validator parameters...")

    with np.load(params_path, allow_pickle=False) as params:
        stats = ValidatorStats(
            class_ids=params["class_ids"].copy(),
            counts=params["counts"].copy(),
            means=params["means"].copy(),
            variances=params["variances"].copy(),
        )
        distance_threshold = float(params["distance_threshold"])
        confidence_threshold = float(params["confidence_threshold"])
        length_limit = float(params["length_limit"])
        epsilon = float(params["epsilon"])

    print("2. Loading frozen Lab 3 model...")
    model, vocabulary = load_frozen_model(lab4.source.checkpoint_path)

    print("3. Reading independent test split...")
    records = list(iter_split_records(lab3, "test"))

    if not records:
        raise ValueError("Test split is empty")

    texts = [record[lab3.data.text_column] for record in records]
    true_labels = np.asarray(
        [int(record[lab3.data.target_column]) for record in records],
        dtype=np.int64,
    )

    print("Test documents:", len(texts))

    print("4. Calculating features and model predictions...")
    features = extract_feature_matrix(
        texts,
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )

    distances = distance_to_training(
        features,
        stats,
        epsilon=epsilon,
    )

    predictions, confidences = predict_with_confidence(
        texts,
        model,
        vocabulary,
    )

    print("5. Applying frozen rejection rules...")
    baseline_rejected = reject_predictions(
        distances,
        confidences,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
    )

    rejected = reject_with_length(
        baseline_rejected,
        texts,
        length_limit,
    )

    accepted = ~rejected
    total = len(texts)
    rejected_count = int(rejected.sum())
    accepted_count = int(accepted.sum())

    false_reject_rate = rejected_count / total
    coverage = accepted_count / total
    model_accuracy = float(np.mean(predictions == true_labels))

    accepted_accuracy = (
        float(np.mean(predictions[accepted] == true_labels[accepted]))
        if accepted_count
        else None
    )

    metrics = {
        "experiment": "independent_id_test",
        "test_rows": total,
        "accepted_rows": accepted_count,
        "rejected_rows": rejected_count,
        "false_reject_rate": false_reject_rate,
        "coverage": coverage,
        "model_accuracy": model_accuracy,
        "accepted_accuracy": accepted_accuracy,
        "distance_threshold": distance_threshold,
        "confidence_threshold": confidence_threshold,
        "length_limit": length_limit,
    }

    print("6. Saving test metrics...")
    output_path = output_dir / "holdout_id_metrics.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics),
        )
        writer.writeheader()
        writer.writerow(metrics)

    print("7. Logging independent test to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(lab4.mlflow.experiment_name)

    with mlflow.start_run(run_name="lab4_holdout_id_test"):
        mlflow.log_params(
            {
                "evaluation": "independent_id_test",
                "distance_threshold": distance_threshold,
                "confidence_threshold": confidence_threshold,
                "length_limit": length_limit,
                "test_rows": total,
            }
        )

        mlflow.log_metrics(
            {
                "false_reject_rate": false_reject_rate,
                "coverage": coverage,
                "model_accuracy": model_accuracy,
                "accepted_accuracy": (
                    accepted_accuracy if accepted_accuracy is not None else 0.0
                ),
            }
        )
        mlflow.log_artifact(str(output_path))

    print()
    print("Test documents:", total)
    print("Accepted:", accepted_count)
    print("Rejected:", rejected_count)
    print("False rejection rate:", round(false_reject_rate, 4))
    print("Coverage:", round(coverage, 4))
    print("Model accuracy:", round(model_accuracy, 4))
    print(
        "Accuracy among accepted:",
        round(accepted_accuracy, 4) if accepted_accuracy is not None else "N/A",
    )

    if false_reject_rate <= lab4.decision.max_false_reject_rate:
        print("PASS: ID false rejection rate <= 10%")
    else:
        print("WARNING: ID false rejection rate > 10%")

    print("Saved:", output_path)
    print("PASS: independent ID evaluation completed")


if __name__ == "__main__":
    main()
