"""Evaluate frozen Lab 4 validator on real foreign-topic documents."""

from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import numpy as np
from sklearn.datasets import fetch_20newsgroups

from src.lab4.config import load_config
from src.lab4.features import extract_feature_matrix
from src.lab4.length_guard import reject_with_length, token_lengths
from src.lab4.model_confidence import (
    load_frozen_model,
    predict_with_confidence,
)
from src.lab4.validator import (
    ValidatorStats,
    distance_to_training,
    reject_predictions,
)

FOREIGN_CATEGORIES = (
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "misc.forsale",
)


def main():
    config = load_config()
    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading frozen validator...")
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
        distance_threshold = float(params["distance_threshold"])
        confidence_threshold = float(params["confidence_threshold"])
        length_limit = float(params["length_limit"])
        epsilon = float(params["epsilon"])

    model, vocabulary = load_frozen_model(config.source.checkpoint_path)

    print("2. Loading unseen categories from 20 Newsgroups...")
    texts = []
    categories = []

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

        if len(selected) < 50:
            raise ValueError(f"Not enough documents for {category}: {len(selected)}")

        texts.extend(selected)
        categories.extend([category] * len(selected))

    print("Foreign documents:", len(texts))

    print("3. Applying frozen model and validator...")
    features = extract_feature_matrix(
        texts,
        vocabulary,
        frequency_bins=config.features.frequency_bins,
        use_oov_ratio=config.features.use_oov_ratio,
    )
    distances = distance_to_training(
        features,
        stats,
        epsilon=epsilon,
    )
    _, confidences = predict_with_confidence(
        texts,
        model,
        vocabulary,
    )

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

    lengths = token_lengths(texts)
    total = len(texts)
    rejected_count = int(rejected.sum())
    accepted_count = total - rejected_count
    recall = rejected_count / total

    print("4. Saving category results...")
    category_rows = []

    for category in FOREIGN_CATEGORIES:
        indices = [i for i, name in enumerate(categories) if name == category]
        detected = int(rejected[indices].sum())
        category_rows.append(
            {
                "category": category,
                "total": len(indices),
                "rejected": detected,
                "accepted": len(indices) - detected,
                "recall": detected / len(indices),
            }
        )

    category_path = output_dir / "holdout_ood_by_category.csv"
    with category_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)

    metrics = {
        "experiment": "independent_real_ood_test",
        "foreign_rows": total,
        "rejected_rows": rejected_count,
        "accepted_rows": accepted_count,
        "foreign_recall": recall,
        "required_recall": config.decision.min_foreign_recall,
        "distance_threshold": distance_threshold,
        "confidence_threshold": confidence_threshold,
        "length_limit": length_limit,
    }

    metrics_path = output_dir / "holdout_ood_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print("5. Logging accepted foreign examples...")
    accepted_path = output_dir / "holdout_ood_accepted.log"
    lines = [
        "Real OOD test: accepted foreign documents",
        "These examples were NOT rejected by the validator.",
        "",
    ]

    for i, is_rejected in enumerate(rejected):
        if not is_rejected:
            lines.append(
                f"index={i}, category={categories[i]}, "
                f"distance={distances[i]:.4f}, "
                f"confidence={confidences[i]:.4f}, "
                f"length={lengths[i]}"
            )

    accepted_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("6. Logging results to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="lab4_holdout_real_ood"):
        mlflow.log_params(
            {
                "evaluation": "independent_real_ood_test",
                "foreign_categories": ",".join(FOREIGN_CATEGORIES),
                "documents_per_category": 50,
            }
        )
        mlflow.log_metrics(
            {
                "foreign_recall": recall,
                "foreign_rejected": rejected_count,
                "foreign_accepted": accepted_count,
            }
        )
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(category_path))
        mlflow.log_artifact(str(accepted_path))

    print()
    print("Foreign documents:", total)
    print("Rejected:", rejected_count)
    print("Accepted:", accepted_count)
    print("Foreign recall:", round(recall, 4))
    print()

    for row in category_rows:
        print(f"{row['category']}: {row['rejected']}/{row['total']} rejected")

    print()
    if recall >= config.decision.min_foreign_recall:
        print("PASS: real OOD recall >= 90%")
    else:
        print("WARNING: real OOD recall < 90%")

    if accepted_count:
        print("WARNING: foreign documents were accepted; see holdout_ood_accepted.log")

    print("Saved:", metrics_path)
    print("Saved:", category_path)
    print("Saved:", accepted_path)
    print("PASS: real OOD evaluation completed")


if __name__ == "__main__":
    main()
