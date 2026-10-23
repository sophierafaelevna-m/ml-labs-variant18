"""Inference confidence from Lab 3's frozen EmbeddingBag+MLP checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.lab3.network import EmbeddingBagMLP
from src.lab3.text_data import collate_embeddingbag, encode


def load_frozen_model(checkpoint_path: str | Path):
    """Restore the exact trained vocabulary and model, on CPU."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    vocabulary = checkpoint["vocabulary"]
    config = checkpoint["model_config"]

    model = EmbeddingBagMLP(
        vocabulary_size=len(vocabulary),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, vocabulary


def predict_with_confidence(
    texts: list[str],
    model: EmbeddingBagMLP,
    vocabulary: dict[str, int],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class IDs and maximum softmax probabilities."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not texts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    predicted_batches = []
    confidence_batches = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            examples = [(encode(text, vocabulary), 0) for text in batch]
            token_ids, offsets, _ = collate_embeddingbag(examples)

            logits = model(token_ids, offsets)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)

            predicted_batches.append(predicted.cpu().numpy())
            confidence_batches.append(confidence.cpu().numpy().astype(np.float64))

    return np.concatenate(predicted_batches), np.concatenate(confidence_batches)
