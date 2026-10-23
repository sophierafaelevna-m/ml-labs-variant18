"""Training-derived length guard for Lab 4."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.lab3.text_data import tokenize


def token_lengths(texts: Iterable[str]) -> np.ndarray:
    """Count tokens using the same tokenizer as Lab 3."""
    return np.asarray(
        [len(tokenize(text)) for text in texts],
        dtype=np.int64,
    )


def fit_length_limit(
    training_texts: Iterable[str],
    quantile: float = 0.99,
) -> float:
    """Fit the upper length limit using training texts only."""
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1")

    lengths = token_lengths(training_texts)

    if lengths.size == 0:
        raise ValueError("Training texts must not be empty")

    return float(np.quantile(lengths, quantile))


def reject_with_length(
    baseline_rejected: np.ndarray,
    texts: list[str],
    length_limit: float,
) -> np.ndarray:
    """Combine existing rejection decisions with the length guard."""
    baseline_rejected = np.asarray(baseline_rejected, dtype=np.bool_)
    lengths = token_lengths(texts)

    if baseline_rejected.ndim != 1:
        raise ValueError("Rejection flags must be one-dimensional")
    if len(baseline_rejected) != len(lengths):
        raise ValueError("Texts and rejection flags must match")
    if not np.isfinite(length_limit) or length_limit <= 0:
        raise ValueError("Length limit must be finite and positive")

    return baseline_rejected | (lengths > length_limit)
