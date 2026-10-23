"""Diagnose TF-IDF similarity to training-only class centroids."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records

FOREIGN_CATEGORIES = (
    "rec.autos",
    "rec.sport.baseball",
    "sci.med",
    "misc.forsale",
)


def main():
    lab3 = load_lab3()

    print("1. Loading training and validation data...")
    train_records = list(iter_split_records(lab3, "train"))
    val_records = list(iter_split_records(lab3, "val"))

    train_texts = [row[lab3.data.text_column] for row in train_records]
    train_labels = np.asarray(
        [int(row[lab3.data.target_column]) for row in train_records]
    )
    val_texts = [row[lab3.data.text_column] for row in val_records]

    print("2. Fitting TF-IDF on training data only...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)

    class_ids = np.unique(train_labels)
    centroids = []

    for class_id in class_ids:
        class_matrix = train_matrix[train_labels == class_id]
        centroid = csr_matrix(np.asarray(class_matrix.mean(axis=0)))
        centroids.append(centroid)

    centroid_matrix = normalize(vstack(centroids))

    def topic_similarity(texts):
        matrix = vectorizer.transform(texts)
        similarities = cosine_similarity(matrix, centroid_matrix)
        return np.max(similarities, axis=1)

    print("3. Calculating similarity for validation data...")
    val_scores = topic_similarity(val_texts)

    print("4. Loading development foreign categories...")
    foreign_texts = []
    foreign_labels = []

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

        if len(selected) != 50:
            raise ValueError(
                f"Expected 50 documents for {category}, got {len(selected)}"
            )

        foreign_texts.extend(selected)
        foreign_labels.extend([category] * len(selected))

    foreign_scores = topic_similarity(foreign_texts)

    print("5. Comparing score distributions...")
    quantiles = [0.1, 0.5, 0.9]

    print()
    print("TRAIN documents:", len(train_texts))
    print("VALIDATION documents:", len(val_texts))
    print("FOREIGN documents:", len(foreign_texts))
    print("TF-IDF features:", len(vectorizer.vocabulary_))
    print()

    print("ID similarity quantiles (10%, 50%, 90%):")
    print(np.round(np.quantile(val_scores, quantiles), 4))

    print("Foreign similarity quantiles (10%, 50%, 90%):")
    print(np.round(np.quantile(foreign_scores, quantiles), 4))

    y_true = np.concatenate(
        [
            np.zeros(len(val_scores), dtype=int),
            np.ones(len(foreign_scores), dtype=int),
        ]
    )
    anomaly_scores = -np.concatenate([val_scores, foreign_scores])
    auc = roc_auc_score(y_true, anomaly_scores)

    print()
    print("Topic similarity ROC-AUC:", round(float(auc), 4))

    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "topic_similarity_diagnostic.csv"

    rows = [
        {"group": "id_validation", "similarity": float(score)} for score in val_scores
    ]
    rows.extend(
        {
            "group": category,
            "similarity": float(score),
        }
        for category, score in zip(foreign_labels, foreign_scores, strict=True)
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "similarity"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_path)
    print("PASS: topic similarity diagnostics completed")


if __name__ == "__main__":
    main()
