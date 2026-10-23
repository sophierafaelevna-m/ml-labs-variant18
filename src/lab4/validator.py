"""Class-conditional NumPy validator with Full and Chunk fitting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidatorStats:
    class_ids: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def _check_batch(features: np.ndarray, labels: np.ndarray) -> None:
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features (n, d) and labels (n,)")
    if features.shape[0] != labels.shape[0] or features.shape[0] == 0:
        raise ValueError("Features and labels must have matching nonzero rows")
    if features.shape[1] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("Features must be nonempty and finite")


def fit_full(features: np.ndarray, labels: np.ndarray) -> ValidatorStats:
    """Compute class statistics with all features resident in memory."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    _check_batch(features, labels)

    class_ids, counts = np.unique(labels, return_counts=True)
    means = np.vstack([features[labels == cls].mean(axis=0) for cls in class_ids])
    variances = np.vstack([features[labels == cls].var(axis=0) for cls in class_ids])
    return ValidatorStats(class_ids, counts, means, variances)


def fit_chunk(batches: Iterable[tuple[np.ndarray, np.ndarray]]) -> ValidatorStats:
    """Merge per-class mean and variance without storing all examples."""
    accumulators: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    feature_count: int | None = None

    for raw_features, raw_labels in batches:
        features = np.asarray(raw_features, dtype=np.float64)
        labels = np.asarray(raw_labels)
        _check_batch(features, labels)

        if feature_count is None:
            feature_count = features.shape[1]
        elif features.shape[1] != feature_count:
            raise ValueError("Inconsistent feature dimensions")

        for cls in np.unique(labels):
            group = features[labels == cls]
            batch_count = len(group)
            batch_mean = group.mean(axis=0)
            batch_m2 = ((group - batch_mean) ** 2).sum(axis=0)
            key = int(cls)

            if key not in accumulators:
                accumulators[key] = (batch_count, batch_mean, batch_m2)
                continue

            count, mean, m2 = accumulators[key]
            total = count + batch_count
            delta = batch_mean - mean
            new_mean = mean + delta * (batch_count / total)
            new_m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
            accumulators[key] = (total, new_mean, new_m2)

    if not accumulators:
        raise ValueError("No training batches supplied")

    class_ids = np.asarray(sorted(accumulators), dtype=np.int64)
    counts = np.asarray([accumulators[c][0] for c in class_ids], dtype=np.int64)
    means = np.vstack([accumulators[c][1] for c in class_ids])
    variances = np.vstack([accumulators[c][2] / accumulators[c][0] for c in class_ids])

    return ValidatorStats(class_ids, counts, means, variances)


def distance_to_training(
    features: np.ndarray,
    stats: ValidatorStats,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return minimum standardized distance to a training class."""
    features = np.asarray(features, dtype=np.float64)

    if features.ndim != 2 or features.shape[1] != stats.means.shape[1]:
        raise ValueError("Incorrect feature matrix shape")
    if not np.all(np.isfinite(features)) or epsilon <= 0:
        raise ValueError("Features must be finite and epsilon positive")

    standardized = (features[:, None, :] - stats.means[None, :, :]) ** 2
    standardized /= stats.variances[None, :, :] + epsilon

    return np.sqrt(standardized.mean(axis=2).min(axis=1))


def reject_predictions(
    distances: np.ndarray,
    confidences: np.ndarray,
    distance_threshold: float,
    confidence_threshold: float,
) -> np.ndarray:
    """Return True for inputs that should be rejected."""
    distances = np.asarray(distances, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)

    if distances.ndim != 1 or confidences.shape != distances.shape:
        raise ValueError("Distances and confidences must have equal lengths")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if not np.all(np.isfinite(confidences)) or np.any(
        (confidences < 0) | (confidences > 1)
    ):
        raise ValueError("Confidences must lie in [0, 1]")
    if distance_threshold <= 0 or not 0 < confidence_threshold < 1:
        raise ValueError("Invalid rejection thresholds")

    return (distances > distance_threshold) | (confidences < confidence_threshold)
