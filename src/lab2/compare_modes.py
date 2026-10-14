import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"

with (OUT / "full_results.json").open(encoding="utf-8") as f:
    full = json.load(f)

with (OUT / "chunk_results.json").open(encoding="utf-8") as f:
    chunk = json.load(f)

rows = []
for mode, result in [("full", full), ("chunk", chunk)]:
    for metric, value in result["metrics"].items():
        if metric.endswith("_ci_lower") or metric.endswith("_ci_upper"):
            continue
        rows.append(
            {
                "mode": mode,
                "metric": metric,
                "value": value,
                "ci_lower": result["metrics"].get(f"{metric}_ci_lower"),
                "ci_upper": result["metrics"].get(f"{metric}_ci_upper"),
                "mlflow_run_id": result.get("mlflow_run_id"),
            }
        )

with (OUT / "ml_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["mode", "metric", "value", "ci_lower", "ci_upper", "mlflow_run_id"],
    )
    writer.writeheader()
    writer.writerows(rows)

memory_rows = []
for mode, result in [("full", full), ("chunk", chunk)]:
    memory_rows.append(
        {
            "mode": mode,
            "training_time_sec": result.get("training_time_sec"),
            "peak_memory_mb": result.get("peak_memory_mb"),
            "mlflow_run_id": result.get("mlflow_run_id"),
        }
    )

with (OUT / "memory_compare.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["mode", "training_time_sec", "peak_memory_mb", "mlflow_run_id"],
    )
    writer.writeheader()
    writer.writerows(memory_rows)

print(f"Created {OUT / 'ml_metrics.csv'}")
print(f"Created {OUT / 'memory_compare.csv'}")
print("\nComparison:")
for row in memory_rows:
    print(row)
