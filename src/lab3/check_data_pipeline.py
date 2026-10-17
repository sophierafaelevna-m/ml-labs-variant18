"""Step 2 smoke test: train-only vocabulary, two loaders, neural network forward."""

from collections import Counter
from itertools import islice

import torch
from torch.utils.data import DataLoader

from src.lab3.config import load_config
from src.lab3.network import EmbeddingBagMLP
from src.lab3.text_data import (
    FullTextDataset,
    StreamingTextDataset,
    build_vocabulary,
    collate_embeddingbag,
    iter_split_records,
)


def main() -> None:
    config = load_config()
    vocabulary = build_vocabulary(config)
    assert vocabulary["<PAD>"] == 0 and vocabulary["<UNK>"] == 1
    print(f"PASS: train-only vocabulary, size={len(vocabulary)}")

    counts: dict[str, Counter[int]] = {}
    for split in ("train", "val", "test"):
        counts[split] = Counter(
            int(row[config.data.target_column])
            for row in iter_split_records(config, split)
        )
        print(
            f"{split}: rows={sum(counts[split].values())}, classes={dict(sorted(counts[split].items()))}"
        )
    assert sum(counts["train"].values()) + sum(counts["val"].values()) == 2034
    assert sum(counts["test"].values()) == 1353
    assert len(counts["train"]) == len(counts["val"]) == len(counts["test"]) == 4

    full = FullTextDataset(config, "train", vocabulary)
    stream = StreamingTextDataset(config, "train", vocabulary)
    assert len(full) == sum(counts["train"].values())
    assert list(islice(iter(stream), 12)) == full.examples[:12]
    print("PASS: first 12 full and streaming examples match")

    batch_size = config.training.batch_size
    full_loader = DataLoader(
        full, batch_size=batch_size, collate_fn=collate_embeddingbag
    )
    stream_loader = DataLoader(
        stream, batch_size=batch_size, collate_fn=collate_embeddingbag
    )
    full_batch = next(iter(full_loader))
    stream_batch = next(iter(stream_loader))
    assert all(torch.equal(a, b) for a, b in zip(full_batch, stream_batch))
    print(f"PASS: Full and Chunk DataLoader batches match, batch_size={batch_size}")

    num_classes = len(counts["train"])
    model = EmbeddingBagMLP(
        vocabulary_size=len(vocabulary),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        num_classes=num_classes,
    )
    with torch.no_grad():
        logits = model(full_batch[0], full_batch[1])
    assert logits.shape == (len(full_batch[2]), num_classes)
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"PASS: neural network forward, output_shape={tuple(logits.shape)}, parameters={parameters}"
    )
    print("PASS: Lab 3 step 2 checks finished")


if __name__ == "__main__":
    main()
