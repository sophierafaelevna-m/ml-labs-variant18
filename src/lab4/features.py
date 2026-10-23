"""Text features for the Lab 4 validator (variant 18)."""

from __future__ import annotations

import numpy as np

from src.lab3.text_data import tokenize


def extract_features(
    text: str,
    vocabulary: dict[str, int],
    frequency_bins: int = 4,
    use_oov_ratio: bool = True,
) -> np.ndarray:
    """Extract length, vocabulary-frequency bins and OOV ratio.

    Vocabulary IDs from Lab 3 are ordered by decreasing token frequency:
    lower IDs correspond to more frequent training tokens.

    The vocabulary must be fitted on training data only.
    """
    if frequency_bins < 2:
        raise ValueError("frequency_bins must be at least 2")
    if len(vocabulary) <= 2:
        raise ValueError("Vocabulary must contain training tokens")

    tokens = tokenize(text)
    token_count = len(tokens)
    bin_counts = np.zeros(frequency_bins, dtype=np.float64)
    unknown_count = 0
    known_vocabulary_size = len(vocabulary) - 2

    for token in tokens:
        token_id = vocabulary.get(token, 1)

        if token_id < 2:
            unknown_count += 1
            continue

        bin_index = min(
            (token_id - 2) * frequency_bins // known_vocabulary_size,
            frequency_bins - 1,
        )
        bin_counts[bin_index] += 1

    denominator = max(token_count, 1)
    relative_bins = bin_counts / denominator
    length_feature = np.log1p(token_count)

    values = [length_feature, *relative_bins.tolist()]

    if use_oov_ratio:
        values.append(unknown_count / denominator)

    result = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite text features detected")

    return result


def extract_feature_matrix(
    texts: list[str],
    vocabulary: dict[str, int],
    frequency_bins: int = 4,
    use_oov_ratio: bool = True,
) -> np.ndarray:
    """Convert a list of texts to a two-dimensional feature matrix."""
    feature_count = 1 + frequency_bins + int(use_oov_ratio)

    if not texts:
        return np.empty((0, feature_count), dtype=np.float64)

    return np.vstack(
        [
            extract_features(
                text,
                vocabulary,
                frequency_bins=frequency_bins,
                use_oov_ratio=use_oov_ratio,
            )
            for text in texts
        ]
    )
