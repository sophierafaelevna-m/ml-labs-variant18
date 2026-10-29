"""Load and verify Lab 3 text classification models for Lab 6."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from src.lab3.network import EmbeddingBagMLP
from src.lab3.text_data import encode


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA256 checksum of a model file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedModel:
    """Verified model and its inference metadata."""

    version: str
    sha256: str
    model: EmbeddingBagMLP
    vocabulary: dict[str, int]
    num_classes: int

    def predict(self, text: str) -> dict:
        """Predict a class for one non-empty text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")

        token_ids = encode(text, self.vocabulary)

        tokens = torch.tensor(token_ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(tokens, offsets)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())

        return {
            "version": self.version,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }


def load_model(
    path: str | Path,
    version: str,
    expected_sha256: str,
) -> LoadedModel:
    """Verify checksum and load a Lab 3 checkpoint."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    actual_sha256 = calculate_sha256(path)

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {version}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    vocabulary = checkpoint["vocabulary"]
    state_dict = checkpoint["model_state_dict"]
    num_classes = int(checkpoint["num_classes"])
    config = checkpoint["model_config"]

    embedding_dim = state_dict["embedding.weight"].shape[1]
    hidden_dim = state_dict["classifier.0.weight"].shape[0]

    model = EmbeddingBagMLP(
        vocabulary_size=len(vocabulary),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=float(config.get("dropout", 0.0)),
        num_classes=num_classes,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return LoadedModel(
        version=version,
        sha256=actual_sha256,
        model=model,
        vocabulary=vocabulary,
        num_classes=num_classes,
    )
