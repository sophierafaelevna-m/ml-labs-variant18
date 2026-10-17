"""Non-destructive Lab 3 validation: repeatability, checkpoints, negative controls, streaming.

Run: python -m src.lab3.verify_lab3
Only writes reports/LAB3/verification_results.json. Does NOT retrain/overwrite the saved models.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError

from src.lab3.config import Lab3Config, load_config
from src.lab3.network import EmbeddingBagMLP
from src.lab3.text_data import build_vocabulary
from src.lab3.train import (
    ChunkTextDataset,
    loader,
    measure_metrics,
    run_epoch,
    seed_everything,
    sha256,
)

OUT = Path("reports/LAB3")


def model_from_checkpoint(checkpoint):
    spec = checkpoint["model_config"]
    model = EmbeddingBagMLP(
        len(checkpoint["vocabulary"]),
        spec["embedding_dim"],
        spec["hidden_dim"],
        spec["dropout"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def train_repeat(cfg, vocab):
    """Same algorithm/ordering as train.py, without MLflow or writing checkpoints."""
    seed_everything(cfg.training.seed)
    model = EmbeddingBagMLP(
        len(vocab), cfg.model.embedding_dim, cfg.model.hidden_dim, cfg.model.dropout, 4
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    train_loader = loader(cfg, "train", vocab, "full")
    val_loader = loader(cfg, "val", vocab, "full")
    best_loss, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        tr = run_epoch(model, train_loader, torch.device("cpu"), optimizer)
        va = run_epoch(model, val_loader, torch.device("cpu"))
        history.append([tr["loss"], va["loss"], tr["macro_f1"], va["macro_f1"]])
        if va["loss"] < best_loss - 1e-6:
            best_loss, best_state, patience = (
                va["loss"],
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    test = run_epoch(model, loader(cfg, "test", vocab, "full"), torch.device("cpu"))
    return model, np.array(history), test


def streaming_peak(cfg, vocab, path, count):
    # Generate outside the measurement: only measure the streaming reader itself.
    with path.open("w", encoding="utf-8") as fh:
        for i in range(count):
            fh.write(
                json.dumps(
                    {
                        "text": "space graphics computer astronomy stars " * 2,
                        "target": i % 4,
                    }
                )
                + "\n"
            )
    data_cfg = cfg.data.model_copy(update={"train_path": path})
    cfg_tmp = cfg.model_copy(update={"data": data_cfg})
    gc.collect()
    tracemalloc.start()
    consumed = sum(1 for _ in ChunkTextDataset(cfg_tmp, "train", vocab))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "source_rows": count,
        "consumed_train_rows": consumed,
        "tracemalloc_peak_mb": round(peak / 1024**2, 5),
    }


def main():
    cfg = load_config()
    vocab = build_vocabulary(cfg)
    result = {"seed": cfg.training.seed, "checks": {}, "warnings": []}
    checks = result["checks"]
    print("Checking saved model files and metrics...", flush=True)
    checkpoints = {}
    for mode in ("full", "chunk"):
        meta_path = OUT / f"training_run_{mode}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint_path = OUT / (
            "dl_model.pt" if mode == "full" else "dl_model_chunk.pt"
        )
        checkpoints[mode] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        actual = sha256(checkpoint_path)
        if actual != meta["checkpoint_sha256"]:
            raise AssertionError(f"{mode}: checkpoint SHA256 mismatch")
        checks[f"{mode}_checkpoint_sha256"] = actual
        print(f"PASS: {mode} checkpoint SHA256", flush=True)

    # Compare all model weights, not only rounded metrics.
    left = checkpoints["full"]["model_state_dict"]
    right = checkpoints["chunk"]["model_state_dict"]
    max_weight_diff = max((left[key] - right[key]).abs().max().item() for key in left)
    checks["full_chunk_max_parameter_difference"] = max_weight_diff
    print(f"Full/Chunk max weight difference: {max_weight_diff:.9g}", flush=True)
    if max_weight_diff > 1e-7:
        result["warnings"].append(
            "Full/Chunk checkpoints have different weights despite matching reported metrics"
        )

    print("Checking repeatability (two complete CPU training runs)...", flush=True)
    m1, h1, t1 = train_repeat(cfg, vocab)
    m2, h2, t2 = train_repeat(cfg, vocab)
    if h1.shape != h2.shape:
        raise AssertionError("Repeated runs stopped at different epochs")
    history_diff = float(np.max(np.abs(h1 - h2)))
    state_diff = max(
        float((p1 - p2).abs().max())
        for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values())
    )
    f1_1 = measure_metrics(
        t1["targets"], t1["predictions"], checkpoints["full"]["rare_class"]
    )["macro_f1"]
    f1_2 = measure_metrics(
        t2["targets"], t2["predictions"], checkpoints["full"]["rare_class"]
    )["macro_f1"]
    checks["repeatability"] = {
        "runs": 2,
        "same_seed": cfg.training.seed,
        "max_history_difference": history_diff,
        "max_weight_difference": state_diff,
        "macro_f1_run_1": f1_1,
        "macro_f1_run_2": f1_2,
        "absolute_f1_difference": abs(f1_1 - f1_2),
        "tolerance": 0.005,
    }
    if abs(f1_1 - f1_2) > 0.005:
        raise AssertionError("Repeatability Macro F1 difference exceeds 0.005")
    print(
        f"PASS: repeatability; Macro F1={f1_1:.6f}/{f1_2:.6f}, "
        f"max weight difference={state_diff:.3g}",
        flush=True,
    )

    # Verify reproduced evaluation against existing report without touching it.
    with (OUT / "dl_metrics.csv").open(newline="", encoding="utf-8") as fh:
        metrics = {
            (r["mode"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh)
        }
    checks["saved_vs_retrained_full_macro_f1_difference"] = abs(
        f1_1 - metrics["full", "macro_f1"]
    )
    if checks["saved_vs_retrained_full_macro_f1_difference"] > 0.005:
        result["warnings"].append(
            "Repeated training differs from saved Full metrics by more than 0.005"
        )

    # Negative controls: strict Pydantic config and no reset of the random seed.
    print("Checking negative controls...", flush=True)
    invalid = cfg.model_dump(mode="python")
    invalid["training"].pop("seed")
    try:
        Lab3Config.model_validate(invalid)
        raise AssertionError("Missing seed was unexpectedly accepted")
    except ValidationError:
        pass
    invalid2 = cfg.model_dump(mode="python")
    invalid2["training"]["batch_size"] = 0
    try:
        Lab3Config.model_validate(invalid2)
        raise AssertionError("batch_size=0 was unexpectedly accepted")
    except ValidationError:
        pass
    # Intentionally do not call seed_everything between these initializations.
    a = EmbeddingBagMLP(
        len(vocab), cfg.model.embedding_dim, cfg.model.hidden_dim, cfg.model.dropout, 4
    )
    b = EmbeddingBagMLP(
        len(vocab), cfg.model.embedding_dim, cfg.model.hidden_dim, cfg.model.dropout, 4
    )
    unseeded_diff = max(
        float((x - y).abs().max())
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )
    checks["negative_controls"] = {
        "missing_seed_rejected": True,
        "zero_batch_size_rejected": True,
        "consecutive_initializations_without_reseeding_differ": unseeded_diff > 0,
        "max_initial_parameter_difference": unseeded_diff,
    }
    if unseeded_diff == 0:
        raise AssertionError("Negative control initializations unexpectedly match")
    print(
        "PASS: invalid configs rejected; no-reseed initialization differs", flush=True
    )

    # Bounded Python allocation with a 10x larger source (not total process RSS).
    print("Checking streaming memory scaling...", flush=True)
    with tempfile.TemporaryDirectory(prefix="lab3_streaming_") as temp:
        small = streaming_peak(cfg, vocab, Path(temp) / "small.jsonl", 1000)
        large = streaming_peak(cfg, vocab, Path(temp) / "large.jsonl", 10000)
    checks["streaming_memory_scaling"] = {
        "small": small,
        "large": large,
        "peak_growth_mb": round(
            large["tracemalloc_peak_mb"] - small["tracemalloc_peak_mb"], 5
        ),
        "note": "tracemalloc measures Python allocations during iteration, not whole-process RSS",
    }
    if large["tracemalloc_peak_mb"] > 3 * small["tracemalloc_peak_mb"] + 1:
        result["warnings"].append(
            "Streaming Python allocation growth exceeded heuristic bound"
        )
    print(
        f"Streaming: 1k={small['tracemalloc_peak_mb']:.3f} MB, "
        f"10k={large['tracemalloc_peak_mb']:.3f} MB",
        flush=True,
    )

    # Device parity on the same saved model, same test batch.
    print("Checking CPU/MPS parity...", flush=True)
    model = model_from_checkpoint(checkpoints["full"])
    token_ids, offsets, _ = next(iter(loader(cfg, "test", vocab, "full")))
    with torch.no_grad():
        cpu_logits = model(token_ids, offsets).detach().cpu()
    parity = {
        "cpu_tested": True,
        "mps_available": torch.backends.mps.is_available(),
        "tolerance_max_abs_diff": 1e-5,
    }
    if parity["mps_available"]:
        try:
            gpu_model = model_from_checkpoint(checkpoints["full"]).to("mps")
            with torch.no_grad():
                gpu_logits = gpu_model(token_ids.to("mps"), offsets.to("mps")).cpu()
            parity["max_abs_difference"] = float((cpu_logits - gpu_logits).abs().max())
            parity["pass"] = parity["max_abs_difference"] <= 1e-5
            if not parity["pass"]:
                result["warnings"].append("CPU/MPS parity does not meet 1e-5 threshold")
            print(
                f"CPU/MPS max diff={parity['max_abs_difference']:.8g}; pass={parity['pass']}",
                flush=True,
            )
        except RuntimeError as exc:
            parity["error"] = str(exc)[:400]
            result["warnings"].append(
                "MPS parity test could not run; see device_parity.error"
            )
            print("WARN: MPS parity test could not run", flush=True)
    else:
        parity["status"] = "not_applicable_mps_unavailable"
        print("SKIP: MPS is unavailable on this computer", flush=True)
    checks["device_parity"] = parity

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "verification_results.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"PASS: saved {path}", flush=True)
    if result["warnings"]:
        print("WARNINGS: " + "; ".join(result["warnings"]), flush=True)


if __name__ == "__main__":
    main()
