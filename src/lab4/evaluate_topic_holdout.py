"""Evaluate frozen TF-IDF guard on ID test and new OOD categories."""

from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score

from src.lab3.config import load_config
from src.lab3.text_data import iter_split_records
from src.lab4.topic_guard import TopicGuard

NEW_FOREIGN_CATEGORIES = (
    "comp.sys.mac.hardware",
    "rec.motorcycles",
    "sci.electronics",
    "rec.sport.hockey",
)


def main() -> None:
    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading frozen TF-IDF guard...")
    guard = TopicGuard.load(output_dir)
    threshold = guard.threshold

    print("2. Loading ID test documents...")
    lab3 = load_config()
    id_texts = [row[lab3.data.text_column] for row in iter_split_records(lab3, "test")]

    print("3. Loading NEW foreign categories...")
    foreign_texts = []
    foreign_categories = []

    for category in NEW_FOREIGN_CATEGORIES:
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

        if len(selected) != 50:
            raise ValueError(
                f"Expected 50 documents for {category}, got {len(selected)}"
            )

        foreign_texts.extend(selected)
        foreign_categories.extend([category] * 50)

    print("4. Calculating similarity...")
    id_scores = guard.similarity(id_texts)
    foreign_scores = guard.similarity(foreign_texts)

    id_rejected = id_scores < threshold
    foreign_rejected = foreign_scores < threshold

    id_count = len(id_texts)
    foreign_count = len(foreign_texts)

    false_rejections = int(id_rejected.sum())
    detected_foreign = int(foreign_rejected.sum())
    accepted_foreign = foreign_count - detected_foreign

    frr = false_rejections / id_count
    recall = detected_foreign / foreign_count

    precision = (
        detected_foreign / (detected_foreign + false_rejections)
        if detected_foreign + false_rejections
        else 0.0
    )

    labels = np.concatenate(
        [
            np.zeros(id_count, dtype=int),
            np.ones(foreign_count, dtype=int),
        ]
    )
    anomaly_scores = -np.concatenate([id_scores, foreign_scores])
    auc = float(roc_auc_score(labels, anomaly_scores))

    print("5. Calculating per-category results...")
    category_rows = []

    for category in NEW_FOREIGN_CATEGORIES:
        indices = [i for i, name in enumerate(foreign_categories) if name == category]
        detected = int(foreign_rejected[indices].sum())

        category_rows.append(
            {
                "category": category,
                "documents": len(indices),
                "rejected": detected,
                "accepted": len(indices) - detected,
                "recall": detected / len(indices),
            }
        )

    metrics = {
        "method": "tfidf_nearest_neighbor",
        "threshold": threshold,
        "id_documents": id_count,
        "id_false_rejections": false_rejections,
        "id_false_reject_rate": frr,
        "foreign_documents": foreign_count,
        "foreign_rejected": detected_foreign,
        "foreign_accepted": accepted_foreign,
        "foreign_recall": recall,
        "rejection_precision": precision,
        "roc_auc": auc,
        "frr_requirement_met": frr <= 0.10,
        "recall_requirement_met": recall >= 0.90,
    }

    print("6. Saving results...")
    metrics_path = output_dir / "topic_holdout_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    category_path = output_dir / "topic_holdout_categories.csv"

    with category_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)

    accepted_path = output_dir / "topic_holdout_accepted.log"
    accepted_lines = [
        "Foreign documents accepted by the frozen topic guard",
        f"Threshold: {threshold:.6f}",
        "",
    ]

    for index, accepted in enumerate(~foreign_rejected):
        if accepted:
            accepted_lines.append(
                f"index={index}, "
                f"category={foreign_categories[index]}, "
                f"similarity={foreign_scores[index]:.6f}"
            )

    accepted_path.write_text(
        "\n".join(accepted_lines) + "\n",
        encoding="utf-8",
    )

    print("7. Logging results to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lab4_variant18")

    with mlflow.start_run(run_name="lab4_topic_guard_holdout"):
        mlflow.log_params(
            {
                "method": "tfidf_nearest_neighbor",
                "threshold": threshold,
                "foreign_categories": ",".join(NEW_FOREIGN_CATEGORIES),
            }
        )
        mlflow.log_metrics(
            {
                "id_false_reject_rate": frr,
                "foreign_recall": recall,
                "rejection_precision": precision,
                "roc_auc": auc,
                "foreign_accepted": accepted_foreign,
            }
        )
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(category_path))
        mlflow.log_artifact(str(accepted_path))

    print()
    print("FINAL TOPIC GUARD EVALUATION")
    print("=" * 48)
    print("Threshold:", round(threshold, 5))
    print()
    print("ID documents:", id_count)
    print("ID rejected:", false_rejections)
    print("ID false rejection rate:", f"{frr:.2%}")
    print()
    print("Foreign documents:", foreign_count)
    print("Foreign rejected:", detected_foreign)
    print("Foreign accepted:", accepted_foreign)
    print("Foreign recall:", f"{recall:.2%}")
    print("Rejection precision:", f"{precision:.2%}")
    print("ROC-AUC:", round(auc, 4))
    print()
    print("BY CATEGORY")

    for row in category_rows:
        print(f"{row['category']}: {row['rejected']}/{row['documents']} rejected")

    print()
    print(
        "FRR REQUIREMENT:",
        "PASS" if frr <= 0.10 else "FAIL",
    )
    print(
        "RECALL REQUIREMENT:",
        "PASS" if recall >= 0.90 else "FAIL",
    )

    print()
    print("Saved:", metrics_path)
    print("Saved:", category_path)
    print("Saved:", accepted_path)
    print("PASS: holdout evaluation completed")


if __name__ == "__main__":
    main()
