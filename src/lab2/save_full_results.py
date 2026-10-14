import json
import re
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "reports" / "LAB2" / "full_run.log"
MODEL_PATH = PROJECT_ROOT / "reports" / "LAB2" / "ml_model.joblib"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "LAB2" / "full_results.json"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    log = LOG_PATH.read_text(encoding="utf-8")

    metrics = {}

    pattern = re.compile(r"(\w+):\s+([0-9.]+)\s+\[([0-9.]+),\s+([0-9.]+)\]")

    for name, value, lower, upper in pattern.findall(log):
        metrics[name] = float(value)
        metrics[f"{name}_ci_lower"] = float(lower)
        metrics[f"{name}_ci_upper"] = float(upper)

    best_c_match = re.search(r"Best C:\s+([0-9.]+)", log)
    best_f1_match = re.search(
        r"Best validation Macro F1:\s+([0-9.]+)",
        log,
    )
    training_time_match = re.search(
        r"Training time:\s+([0-9.]+)\s+sec",
        log,
    )
    baseline_accuracy_match = re.search(
        r"Baseline accuracy:\s+([0-9.]+)",
        log,
    )
    baseline_f1_match = re.search(
        r"Baseline Macro F1:\s+([0-9.]+)",
        log,
    )

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

    experiment = mlflow.get_experiment_by_name("lab2_variant18")

    if experiment is None:
        raise RuntimeError("MLflow experiment lab2_variant18 not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError("No MLflow runs found.")

    run = runs.iloc[0]

    results = {
        "mode": "full",
        "best_C": (float(best_c_match.group(1)) if best_c_match else None),
        "best_validation_macro_f1": (
            float(best_f1_match.group(1)) if best_f1_match else None
        ),
        "training_time_sec": (
            float(training_time_match.group(1)) if training_time_match else None
        ),
        "baseline": {
            "accuracy": (
                float(baseline_accuracy_match.group(1))
                if baseline_accuracy_match
                else None
            ),
            "macro_f1": (
                float(baseline_f1_match.group(1)) if baseline_f1_match else None
            ),
        },
        "metrics": metrics,
        "mlflow_run_id": run["run_id"],
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256_file(MODEL_PATH),
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"MLflow run: {run['run_id']}")
    print(f"Model SHA-256: {results['model_sha256']}")


if __name__ == "__main__":
    main()
