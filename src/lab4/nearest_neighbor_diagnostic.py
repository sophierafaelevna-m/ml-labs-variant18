"""Compare TF-IDF centroids with nearest training documents."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from src.lab3.config import load_config
from src.lab3.text_data import iter_split_records

CATEGORIES = (
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "misc.forsale",
)


def main():
    lab3 = load_config()
    output_dir = Path("reports/LAB4")

    print("1. Loading train and validation documents...")
    train = [row[lab3.data.text_column] for row in iter_split_records(lab3, "train")]
    validation = [row[lab3.data.text_column] for row in iter_split_records(lab3, "val")]

    print("2. Loading development foreign documents...")
    foreign = []

    for category in CATEGORIES:
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
        foreign.extend(selected)

    print("3. Fitting TF-IDF on train only...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )

    train_matrix = vectorizer.fit_transform(train)

    def nearest_scores(texts):
        query_matrix = vectorizer.transform(texts)
        similarities = query_matrix @ train_matrix.T
        return np.asarray(similarities.max(axis=1).toarray()).ravel()

    print("4. Calculating nearest-neighbor similarity...")
    id_nn = nearest_scores(validation)
    foreign_nn = nearest_scores(foreign)

    print("5. Loading previous centroid scores...")
    with (output_dir / "topic_similarity_diagnostic.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    id_centroid = np.asarray(
        [float(row["similarity"]) for row in rows if row["group"] == "id_validation"]
    )
    foreign_centroid = np.asarray(
        [float(row["similarity"]) for row in rows if row["group"] != "id_validation"]
    )

    assert len(id_nn) == len(id_centroid) == 410
    assert len(foreign_nn) == len(foreign_centroid) == 200

    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(id_nn))
    calibration = permutation[:205]
    validation_check = permutation[205:]

    print()
    print("COMPARISON OF TOPIC SIMILARITY METHODS")
    print("=" * 58)

    for name, id_scores, foreign_scores in (
        ("Centroids", id_centroid, foreign_centroid),
        ("Nearest neighbor", id_nn, foreign_nn),
    ):
        labels = np.concatenate(
            [
                np.zeros(len(id_scores)),
                np.ones(len(foreign_scores)),
            ]
        )
        anomaly = -np.concatenate([id_scores, foreign_scores])
        auc = roc_auc_score(labels, anomaly)

        print()
        print("Method:", name)
        print("ROC-AUC:", round(float(auc), 4))
        print(
            "ID median:",
            round(float(np.median(id_scores)), 4),
        )
        print(
            "Foreign median:",
            round(float(np.median(foreign_scores)), 4),
        )

        for budget in (0.05, 0.08, 0.10):
            threshold = float(np.quantile(id_scores[calibration], budget))
            calibration_frr = float((id_scores[calibration] < threshold).mean())
            check_frr = float((id_scores[validation_check] < threshold).mean())
            foreign_recall = float((foreign_scores < threshold).mean())

            print(
                f"Budget {budget:.0%}: "
                f"threshold={threshold:.5f}, "
                f"calibration_FRR={calibration_frr:.2%}, "
                f"validation_FRR={check_frr:.2%}, "
                f"foreign_recall={foreign_recall:.2%}"
            )

    print()
    print("PASS: nearest-neighbor comparison completed")


if __name__ == "__main__":
    main()
