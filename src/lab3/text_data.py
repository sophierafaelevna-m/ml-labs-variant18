"""Text preparation for variant 18: train-only vocabulary, full and streaming modes.

The validation split is deterministic and independent of Python's randomized hash().
Streaming mode reads one JSONL record at a time without storing the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset

from src.lab3.config import Lab3Config

PAD_ID = 0
UNK_ID = 1
_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def is_validation_row(row_number: int, seed: int, fraction: float) -> bool:
    """Stable train/validation assignment based on row number and seed."""
    digest = hashlib.sha256(f"{seed}:{row_number}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big") / (1 << 64)
    return value < fraction


def iter_records(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if line.strip():
                yield row_number, json.loads(line)


def iter_split_records(config: Lab3Config, split: str) -> Iterator[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    path = config.data.test_path if split == "test" else config.data.train_path
    for row_number, record in iter_records(path):
        if split != "test":
            is_val = is_validation_row(
                row_number, config.training.seed, config.data.validation_fraction
            )
            if (split == "val") != is_val:
                continue
        yield record


def build_vocabulary(config: Lab3Config) -> dict[str, int]:
    """Fit vocabulary ONLY on training rows (not validation or test)."""
    counter: Counter[str] = Counter()
    for record in iter_split_records(config, "train"):
        counter.update(tokenize(record[config.data.text_column]))
    ordered = sorted(
        (
            token
            for token, count in counter.items()
            if count >= config.model.min_token_frequency
        ),
        key=lambda token: (-counter[token], token),
    )[: config.model.vocab_size - 2]
    return {
        "<PAD>": PAD_ID,
        "<UNK>": UNK_ID,
        **{token: i + 2 for i, token in enumerate(ordered)},
    }


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    tokens = tokenize(text)
    return [vocabulary.get(token, UNK_ID) for token in tokens] or [UNK_ID]


def _encoded_records(
    config: Lab3Config, split: str, vocabulary: dict[str, int]
) -> Iterator[tuple[list[int], int]]:
    for record in iter_split_records(config, split):
        yield (
            encode(record[config.data.text_column], vocabulary),
            int(record[config.data.target_column]),
        )


class FullTextDataset(Dataset):
    """Holds all encoded examples for a split in RAM."""

    def __init__(self, config: Lab3Config, split: str, vocabulary: dict[str, int]):
        self.examples = list(_encoded_records(config, split, vocabulary))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


class StreamingTextDataset(IterableDataset):
    """Opens JSONL afresh on each iteration; never materializes the full split."""

    def __init__(self, config: Lab3Config, split: str, vocabulary: dict[str, int]):
        self.config = config
        self.split = split
        self.vocabulary = vocabulary

    def __iter__(self) -> Iterator[tuple[list[int], int]]:
        return _encoded_records(self.config, self.split, self.vocabulary)


def collate_embeddingbag(
    examples: Sequence[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten token IDs and compute offsets for torch.nn.EmbeddingBag."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    flat_ids: list[int] = []
    offsets: list[int] = []
    targets: list[int] = []
    for token_ids, target in examples:
        offsets.append(len(flat_ids))
        flat_ids.extend(token_ids or [UNK_ID])
        targets.append(target)
    return (
        torch.tensor(flat_ids, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
