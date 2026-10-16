"""Lab 3: train identical EmbeddingBag+MLP networks using Full or Chunk loading.

Run separately for each mode to measure process memory independently:
  python -m src.lab3.train --mode full
  python -m src.lab3.train --mode chunk
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from src.lab3.config import load_config
from src.lab3.network import EmbeddingBagMLP
from src.lab3.text_data import (
    FullTextDataset,
    build_vocabulary,
    collate_embeddingbag,
    encode,
    iter_split_records,
)

OUT = Path("reports/LAB3")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Keep CPU execution reproducible across repeated runs on the same machine.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def get_device(name: str) -> torch.device:
    # CPU is deliberately the reproducible default; MPS parity is tested separately.
    if name in ("auto", "cpu"):
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("MPS requested but unavailable; set training.device to cpu")


def memory_mb() -> float:
    rss = psutil.Process(os.getpid()).memory_info().rss
    # macOS ru_maxrss is bytes; Linux ru_maxrss is KiB.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = maxrss if sys.platform == "darwin" else maxrss * 1024
    return max(rss, peak) / (1024 * 1024)


class ChunkTextDataset(IterableDataset):
    """Read and encode at most chunk_size JSONL records at a time from disk."""

    def __init__(self, config, split, vocabulary):
        self.config = config
        self.split = split
        self.vocabulary = vocabulary

    def __iter__(self):
        source = iter_split_records(self.config, self.split)
        while True:
            chunk = list(islice(source, self.config.loading.chunk_size))
            if not chunk:
                break
            for record in chunk:
                yield (
                    encode(record[self.config.data.text_column], self.vocabulary),
                    int(record[self.config.data.target_column]),
                )


def loader(config, split, vocabulary, mode):
    ds = (
        FullTextDataset(config, split, vocabulary)
        if mode == "full"
        else ChunkTextDataset(config, split, vocabulary)
    )
    # Both loaders use the same deterministic order, to isolate the loading mode.
    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_embeddingbag,
    )


def run_epoch(model, data_loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    ys, preds = [], []
    loss_fn = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for token_ids, offsets, targets in data_loader:
            token_ids = token_ids.to(device)
            offsets = offsets.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(token_ids, offsets)
            loss = loss_fn(scores, targets)
            if training:
                loss.backward()
                optimizer.step()
            n = targets.size(0)
            total_loss += loss.item() * n
            total_count += n
            ys.extend(targets.cpu().tolist())
            preds.extend(scores.argmax(1).cpu().tolist())
    if total_count == 0:
        raise ValueError("Empty dataset")
    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "targets": ys,
        "predictions": preds,
    }


def measure_metrics(ys, preds, rare_class):
    y = np.asarray(ys)
    p = np.asarray(preds)
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "macro_precision": precision_score(y, p, average="macro", zero_division=0),
        "macro_recall": recall_score(y, p, average="macro", zero_division=0),
        "rare_class_recall": recall_score(
            y == rare_class, p == rare_class, zero_division=0
        ),
    }


def bootstrap_intervals(ys, preds, rare_class, seed, repeats=300):
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys)
    preds = np.asarray(preds)
    values = {key: [] for key in measure_metrics(ys, preds, rare_class)}
    for _ in range(repeats):
        idx = rng.integers(0, len(ys), size=len(ys))
        scores = measure_metrics(ys[idx], preds[idx], rare_class)
        for name, score in scores.items():
            values[name].append(float(score))
    return {
        name: tuple(np.quantile(scores, [0.025, 0.975]).tolist())
        for name, scores in values.items()
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replace_mode_rows(path, mode, rows, fieldnames):
    old = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            old = [row for row in csv.DictReader(handle) if row["mode"] != mode]
    write_csv(path, fieldnames, old + rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Lab 3 neural network training")
    parser.add_argument("--mode", choices=("full", "chunk"), required=True)
    parser.add_argument("--config", default="configs/lab3.yaml")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Local smoke test only"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode not in cfg.loading.modes:
        raise ValueError(f"Mode {args.mode} is not enabled by configuration")
    seed_everything(cfg.training.seed)
    device = get_device(cfg.training.device)
    OUT.mkdir(parents=True, exist_ok=True)
    vocabulary = build_vocabulary(cfg)
    train_loader = loader(cfg, "train", vocabulary, args.mode)
    val_loader = loader(cfg, "val", vocabulary, args.mode)
    test_loader = loader(cfg, "test", vocabulary, args.mode)
    class_counts = Counter(
        int(row[cfg.data.target_column]) for row in iter_split_records(cfg, "train")
    )
    rare_class = min(class_counts, key=lambda label: (class_counts[label], label))
    num_classes = len(class_counts)
    if set(class_counts) != set(range(num_classes)):
        raise ValueError("Expected contiguous class labels 0..K-1")
    model = EmbeddingBagMLP(
        len(vocabulary),
        cfg.model.embedding_dim,
        cfg.model.hidden_dim,
        cfg.model.dropout,
        num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"Mode={args.mode} device={device} vocab={len(vocabulary)} parameters={parameters}",
        flush=True,
    )

    mlflow = None
    run_id = ""
    if not args.no_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run(run_name=f"lab3_{args.mode}_seed{cfg.training.seed}")
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(
            {
                "mode": args.mode,
                "seed": cfg.training.seed,
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "learning_rate": cfg.training.learning_rate,
                "early_stopping_patience": cfg.training.early_stopping_patience,
                "embedding_dim": cfg.model.embedding_dim,
                "hidden_dim": cfg.model.hidden_dim,
                "dropout": cfg.model.dropout,
                "vocabulary_size": len(vocabulary),
                "parameters": parameters,
                "device": str(device),
                "data_loading_order": "sequential",
                "chunk_size_configured": cfg.loading.chunk_size,
            }
        )

    peak_mb = memory_mb()
    start = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    try:
        for epoch in range(1, cfg.training.epochs + 1):
            train_stats = run_epoch(model, train_loader, device, optimizer)
            val_stats = run_epoch(model, val_loader, device)
            peak_mb = max(peak_mb, memory_mb())
            history.append(
                {
                    "mode": args.mode,
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "val_loss": val_stats["loss"],
                    "train_macro_f1": train_stats["macro_f1"],
                    "val_macro_f1": val_stats["macro_f1"],
                }
            )
            print(
                f"Epoch {epoch:02d}: train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_macro_f1={val_stats['macro_f1']:.4f}",
                flush=True,
            )
            if mlflow:
                for key in ("train_loss", "val_loss", "train_macro_f1", "val_macro_f1"):
                    mlflow.log_metric(key, history[-1][key], step=epoch)
            if val_stats["loss"] < best_loss - 1e-6:
                best_loss = val_stats["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= cfg.training.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}; best epoch {best_epoch}",
                        flush=True,
                    )
                    break
        if best_state is None:
            raise RuntimeError("No best model was selected")
        model.load_state_dict(best_state)
        # Test set is evaluated only after selecting best epoch on validation.
        test_stats = run_epoch(model, test_loader, device)
        elapsed = time.perf_counter() - start
        peak_mb = max(peak_mb, memory_mb())
        point = measure_metrics(
            test_stats["targets"], test_stats["predictions"], rare_class
        )
        intervals = bootstrap_intervals(
            test_stats["targets"],
            test_stats["predictions"],
            rare_class,
            cfg.training.seed + 7,
        )
        checkpoint_path = OUT / (
            "dl_model.pt" if args.mode == "full" else "dl_model_chunk.pt"
        )
        torch.save(
            {
                "model_state_dict": best_state,
                "vocabulary": vocabulary,
                "model_config": cfg.model.model_dump(mode="json"),
                "training_config": cfg.training.model_dump(mode="json"),
                "num_classes": num_classes,
                "rare_class": rare_class,
                "mode": args.mode,
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )
        checkpoint_hash = sha256(checkpoint_path)
        history_path = OUT / f"training_history_{args.mode}.csv"
        write_csv(history_path, list(history[0]), history)
        if mlflow:
            for key, val in point.items():
                mlflow.log_metric(f"test_{key}", val)
            mlflow.log_metric("training_time_sec", elapsed)
            mlflow.log_metric("peak_memory_mb", peak_mb)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_artifact(str(checkpoint_path))
            mlflow.log_artifact(str(history_path))
        metric_rows = [
            {
                "mode": args.mode,
                "metric": name,
                "value": value,
                "ci_low": intervals[name][0],
                "ci_high": intervals[name][1],
                "mlflow_run_id": run_id,
            }
            for name, value in point.items()
        ]
        replace_mode_rows(
            OUT / "dl_metrics.csv",
            args.mode,
            metric_rows,
            ["mode", "metric", "value", "ci_low", "ci_high", "mlflow_run_id"],
        )
        replace_mode_rows(
            OUT / "memory_compare.csv",
            args.mode,
            [
                {
                    "mode": args.mode,
                    "training_time_sec": elapsed,
                    "peak_memory_mb": peak_mb,
                    "epochs_trained": len(history),
                    "best_epoch": best_epoch,
                    "mlflow_run_id": run_id,
                }
            ],
            [
                "mode",
                "training_time_sec",
                "peak_memory_mb",
                "epochs_trained",
                "best_epoch",
                "mlflow_run_id",
            ],
        )
        metadata = {
            "mode": args.mode,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "mlflow_run_id": run_id,
            "seed": cfg.training.seed,
            "device": str(device),
            "parameters": parameters,
            "rare_class": rare_class,
            "best_epoch": best_epoch,
            "test_evaluated_after_validation_selection": True,
        }
        (OUT / f"training_run_{args.mode}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"PASS: mode={args.mode} test_macro_f1={point['macro_f1']:.4f} "
            f"peak_memory_mb={peak_mb:.1f} time_sec={elapsed:.2f}",
            flush=True,
        )
        print(f"Checkpoint: {checkpoint_path} sha256={checkpoint_hash}", flush=True)
        print(f"MLflow run ID: {run_id or 'disabled (smoke test)'}", flush=True)
    finally:
        if mlflow and mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()
