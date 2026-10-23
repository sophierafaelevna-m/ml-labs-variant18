"""Train-only TF-IDF nearest-neighbor applicability guard."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.lab3.config import load_config
from src.lab3.text_data import iter_split_records


@dataclass
class TopicGuard:
    vectorizer: TfidfVectorizer
    train_matrix: object
    threshold: float

    def similarity(self, texts: list[str]) -> np.ndarray:
        """Maximum cosine similarity to training documents."""
        query_matrix = self.vectorizer.transform(texts)
        similarities = query_matrix @ self.train_matrix.T
        return np.asarray(similarities.max(axis=1).toarray()).ravel()

    def reject(self, texts: list[str]) -> np.ndarray:
        """True means that the document is rejected."""
        return self.similarity(texts) < self.threshold

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.vectorizer,
            directory / "topic_vectorizer.joblib",
        )
        save_npz(
            directory / "topic_train_matrix.npz",
            self.train_matrix,
        )

        metadata = {
            "method": "tfidf_nearest_neighbor",
            "threshold": self.threshold,
            "train_documents": self.train_matrix.shape[0],
            "tfidf_features": self.train_matrix.shape[1],
            "calibration_quantile": 0.05,
            "fit_split": "train",
            "calibration_split": "validation_first_half_seed42",
        }

        (directory / "topic_guard_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "TopicGuard":
        metadata = json.loads(
            (directory / "topic_guard_metadata.json").read_text(encoding="utf-8")
        )
        return cls(
            vectorizer=joblib.load(directory / "topic_vectorizer.joblib"),
            train_matrix=load_npz(directory / "topic_train_matrix.npz"),
            threshold=float(metadata["threshold"]),
        )


def main() -> None:
    lab3 = load_config()
    output_dir = Path("reports/LAB4")

    print("1. Loading train and validation data...")
    train_texts = [
        row[lab3.data.text_column] for row in iter_split_records(lab3, "train")
    ]
    val_texts = [row[lab3.data.text_column] for row in iter_split_records(lab3, "val")]

    print("2. Fitting TF-IDF on training documents...")
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

    guard = TopicGuard(
        vectorizer=vectorizer,
        train_matrix=train_matrix,
        threshold=0.0,
    )

    print("3. Computing validation similarities...")
    scores = guard.similarity(val_texts)

    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(val_texts))
    calibration_indices = permutation[: len(val_texts) // 2]
    check_indices = permutation[len(val_texts) // 2 :]

    print("4. Calibrating threshold on 205 documents...")
    guard.threshold = float(np.quantile(scores[calibration_indices], 0.05))

    calibration_frr = float((scores[calibration_indices] < guard.threshold).mean())
    check_frr = float((scores[check_indices] < guard.threshold).mean())

    print("5. Saving fitted guard...")
    guard.save(output_dir)

    print("6. Verifying that saved guard works...")
    restored = TopicGuard.load(output_dir)
    restored_scores = restored.similarity(val_texts[:10])

    np.testing.assert_allclose(
        restored_scores,
        scores[:10],
        rtol=1e-10,
        atol=1e-12,
    )

    metrics = {
        "method": "tfidf_nearest_neighbor",
        "train_documents": len(train_texts),
        "calibration_documents": len(calibration_indices),
        "validation_check_documents": len(check_indices),
        "threshold": guard.threshold,
        "calibration_frr": calibration_frr,
        "validation_check_frr": check_frr,
    }

    metrics_path = output_dir / "topic_guard_calibration.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print()
    print("TRAIN DOCUMENTS:", len(train_texts))
    print("CALIBRATION DOCUMENTS:", len(calibration_indices))
    print("VALIDATION CHECK DOCUMENTS:", len(check_indices))
    print("TF-IDF FEATURES:", train_matrix.shape[1])
    print("TOPIC THRESHOLD:", round(guard.threshold, 5))
    print("CALIBRATION FRR:", round(calibration_frr, 4))
    print("VALIDATION CHECK FRR:", round(check_frr, 4))
    print()
    print("Saved:", metrics_path)
    print("PASS: saved topic guard reproduces predictions")


if __name__ == "__main__":
    main()
