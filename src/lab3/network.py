"""EmbeddingBag + MLP text classifier for variant 18."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingBagMLP(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocabulary_size, embedding_dim, mode="mean", padding_idx=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        features = self.embedding(token_ids, offsets)
        return self.classifier(features)
