"""Check Full/Chunk validator statistics on real 20 Newsgroups data."""

from __future__ import annotations

from itertools import islice

import numpy as np

from src.lab3.config import load_config as load_lab3_config
from src.lab3.text_data import build_vocabulary, iter_split_records
from src.lab4.config import load_config as load_lab4_config
from src.lab4.features import extract_feature_matrix
from src.lab4.validator import distance_to_training, fit_chunk, fit_full


def make_feature_batches(lab3, lab4, vocabulary, split):
    """Read JSONL incrementally and yield feature/label batches."""
    records = iter(iter_split_records(lab3, split))
    text_column = lab3.data.text_column
    target_column = lab3.data.target_column

    while True:
        batch = list(islice(records, lab4.loading.chunk_size))
        if not batch:
            break

        features = extract_feature_matrix(
            [record[text_column] for record in batch],
            vocabulary,
            frequency_bins=lab4.features.frequency_bins,
            use_oov_ratio=lab4.features.use_oov_ratio,
        )
        labels = np.asarray(
            [int(record[target_column]) for record in batch],
            dtype=np.int64,
        )
        yield features, labels


def main():
    lab3 = load_lab3_config("configs/lab3.yaml")
    lab4 = load_lab4_config("configs/lab4.yaml")

    print("Building training-only vocabulary...")
    vocabulary = build_vocabulary(lab3)
    print("Vocabulary size:", len(vocabulary))

    print("\n1. Fitting Full validator...")
    train_records = list(iter_split_records(lab3, "train"))
    full_features = extract_feature_matrix(
        [record[lab3.data.text_column] for record in train_records],
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )
    full_labels = np.asarray(
        [int(record[lab3.data.target_column]) for record in train_records],
        dtype=np.int64,
    )
    full_stats = fit_full(full_features, full_labels)

    print("Training rows:", len(train_records))
    print("Feature matrix shape:", full_features.shape)
    print("Class IDs:", full_stats.class_ids.tolist())
    print("Class counts:", full_stats.counts.tolist())

    print("\n2. Fitting Chunk validator...")
    chunk_stats = fit_chunk(make_feature_batches(lab3, lab4, vocabulary, "train"))

    np.testing.assert_array_equal(full_stats.class_ids, chunk_stats.class_ids)
    np.testing.assert_array_equal(full_stats.counts, chunk_stats.counts)
    np.testing.assert_allclose(full_stats.means, chunk_stats.means, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        full_stats.variances, chunk_stats.variances, rtol=0, atol=1e-10
    )

    mean_diff = np.max(np.abs(full_stats.means - chunk_stats.means))
    variance_diff = np.max(np.abs(full_stats.variances - chunk_stats.variances))

    print("Max mean difference:", mean_diff)
    print("Max variance difference:", variance_diff)
    print("PASS: Full and Chunk statistics match on real data")

    print("\n3. Checking held-out validation texts...")
    validation_distances = []

    for features, _ in make_feature_batches(lab3, lab4, vocabulary, "val"):
        distances = distance_to_training(
            features,
            full_stats,
            epsilon=lab4.distance.epsilon,
        )
        validation_distances.append(distances)

    if not validation_distances:
        raise ValueError("Validation split is empty")

    distances = np.concatenate(validation_distances)
    if not np.all(np.isfinite(distances)):
        raise ValueError("Non-finite validation distances")

    print("Validation rows:", len(distances))
    print(
        "Distance quantiles (50%, 90%, 95%, 99%):",
        np.round(np.quantile(distances, [0.5, 0.9, 0.95, 0.99]), 4),
    )
    print(
        "Share above initial distance threshold:",
        round(float(np.mean(distances > lab4.distance.threshold)), 4),
    )
    print("PASS: held-out validation distances calculated")
    print("\nPASS: Lab 4 real-data checks finished")


if __name__ == "__main__":
    main()
