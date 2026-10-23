"""Benchmark Lab 4 Full/Chunk statistics and log results to MLflow."""

from __future__ import annotations

import csv
import gc
import json
import time
import tracemalloc
from pathlib import Path

import mlflow
import numpy as np

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.check_real_data import make_feature_batches
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.model_confidence import load_frozen_model
from src.lab4.validator import fit_chunk, fit_full


def benchmark(operation):
    """Measure elapsed time and peak Python-tracked allocations."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()

    try:
        stats = operation()
        elapsed = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return stats, {
        "time_seconds": elapsed,
        "peak_traced_memory_mib": peak_bytes / (1024**2),
    }


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()
    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, vocabulary = load_frozen_model(lab4.source.checkpoint_path)

    def run_full():
        records = list(iter_split_records(lab3, "train"))
        features = extract_feature_matrix(
            [row[lab3.data.text_column] for row in records],
            vocabulary,
            frequency_bins=lab4.features.frequency_bins,
            use_oov_ratio=lab4.features.use_oov_ratio,
        )
        labels = np.asarray(
            [int(row[lab3.data.target_column]) for row in records],
            dtype=np.int64,
        )
        return fit_full(features, labels)

    def run_chunk():
        return fit_chunk(make_feature_batches(lab3, lab4, vocabulary, "train"))

    print("1. Benchmarking Full...")
    full_stats, full_metrics = benchmark(run_full)

    print("2. Benchmarking Chunk...")
    chunk_stats, chunk_metrics = benchmark(run_chunk)

    print("3. Verifying numerical equivalence...")
    np.testing.assert_array_equal(full_stats.class_ids, chunk_stats.class_ids)
    np.testing.assert_array_equal(full_stats.counts, chunk_stats.counts)
    np.testing.assert_allclose(full_stats.means, chunk_stats.means, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        full_stats.variances, chunk_stats.variances, rtol=0, atol=1e-10
    )

    mean_difference = float(np.max(np.abs(full_stats.means - chunk_stats.means)))
    variance_difference = float(
        np.max(np.abs(full_stats.variances - chunk_stats.variances))
    )

    print("PASS: Full and Chunk statistics match")

    rows = [
        {"mode": "full", **full_metrics},
        {"mode": "chunk", **chunk_metrics},
    ]

    comparison_path = output_dir / "full_chunk_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "full": full_metrics,
        "chunk": chunk_metrics,
        "max_mean_difference": mean_difference,
        "max_variance_difference": variance_difference,
        "allclose_atol": 1e-10,
        "memory_measurement": ("Peak tracemalloc-tracked allocations, not process RSS"),
    }

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("4. Logging experiments to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(lab4.mlflow.experiment_name)

    for row in rows:
        with mlflow.start_run(run_name=f"lab4_{row['mode']}"):
            mlflow.log_params(
                {
                    "mode": row["mode"],
                    "chunk_size": lab4.loading.chunk_size,
                    "frequency_bins": lab4.features.frequency_bins,
                    "vocabulary_size": len(vocabulary),
                    "training_rows": int(full_stats.counts.sum()),
                }
            )
            mlflow.log_metrics(
                {
                    "time_seconds": row["time_seconds"],
                    "peak_traced_memory_mib": row["peak_traced_memory_mib"],
                    "max_mean_difference": mean_difference,
                    "max_variance_difference": variance_difference,
                }
            )
            mlflow.log_artifact(str(comparison_path))
            mlflow.log_artifact(str(summary_path))

    print()
    for row in rows:
        print(
            f"{row['mode'].upper()}: "
            f"time={row['time_seconds']:.4f}s, "
            f"peak_traced_memory="
            f"{row['peak_traced_memory_mib']:.4f} MiB"
        )

    print("Max mean difference:", mean_difference)
    print("Max variance difference:", variance_difference)
    print("Saved:", comparison_path)
    print("Saved:", summary_path)
    print("MLflow tracking database: mlflow.db")
    print("PASS: benchmark and MLflow logging completed")


if __name__ == "__main__":
    main()
