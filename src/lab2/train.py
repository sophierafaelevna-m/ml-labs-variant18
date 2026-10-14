import hashlib
import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np

from src.lab2.config import load_config
from src.lab2.data import load_jsonl, split_train_validation
from src.lab2.metrics import calculate_metrics, naive_baseline
from src.lab2.model import train_final_model, tune_hyperparameter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "lab2.yaml"
MODEL_PATH = PROJECT_ROOT / "reports" / "LAB2" / "ml_model.joblib"


def bootstrap_ci(
    y_true,
    y_pred,
    metric_name: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
):
    """Calculate percentile bootstrap confidence interval."""
    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(y_true),
            size=len(y_true),
        )

        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        metrics = calculate_metrics(
            sample_true,
            sample_pred,
        )

        values.append(metrics[metric_name])

    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))

    return lower, upper


def calculate_test_metrics_with_ci(y_true, y_pred):
    """Calculate test metrics and bootstrap confidence intervals."""
    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    metric_names = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "rare_class_recall",
    ]

    result = {}

    for name in metric_names:
        lower, upper = bootstrap_ci(
            y_true,
            y_pred,
            name,
        )

        result[name] = metrics[name]
        result[f"{name}_ci_lower"] = lower
        result[f"{name}_ci_upper"] = upper

    return result


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print("LAB 2 — FULL MODE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    print("\n[1/7] Loading data...")

    train_df = load_jsonl(PROJECT_ROOT / config.data.train_path)

    test_df = load_jsonl(PROJECT_ROOT / config.data.test_path)

    train_part, validation_part = split_train_validation(
        train_df,
        target_column=config.data.target_column,
        random_state=config.random_state,
    )

    print(f"Full train: {len(train_df)}")
    print(f"Train part: {len(train_part)}")
    print(f"Validation: {len(validation_part)}")
    print(f"Test: {len(test_df)}")

    # ---------------------------------------------------------
    # 2. Hyperparameter tuning
    # ---------------------------------------------------------
    print("\n[2/7] Hyperparameter tuning...")

    tuning_result = tune_hyperparameter(
        train_df=train_part,
        validation_df=validation_part,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
        c_values=[0.1, 1.0, 10.0],
    )

    best_c = tuning_result["best_c"]

    print("\nTuning results:")

    for result in tuning_result["all_results"]:
        print(f"C={result['C']}: macro_f1={result['macro_f1']:.4f}")

    print(f"\nBest C: {best_c}")
    print(f"Best validation Macro F1: {tuning_result['best_score']:.4f}")

    # ---------------------------------------------------------
    # 3. Final training
    # ---------------------------------------------------------
    print("\n[3/7] Training final model...")

    start_time = time.perf_counter()

    model = train_final_model(
        train_df=train_df,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
        c_value=best_c,
    )

    training_time = time.perf_counter() - start_time

    print(f"Training time: {training_time:.3f} sec")

    # ---------------------------------------------------------
    # 4. Test evaluation
    # ---------------------------------------------------------
    print("\n[4/7] Evaluating on test set...")

    y_test = test_df[config.data.target_column]
    predictions = model.predict(test_df[config.data.text_column])

    metrics = calculate_test_metrics_with_ci(
        y_test,
        predictions,
    )

    print("\nTest metrics:")

    for name in [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "rare_class_recall",
    ]:
        print(
            f"{name}: "
            f"{metrics[name]:.4f} "
            f"[{metrics[f'{name}_ci_lower']:.4f}, "
            f"{metrics[f'{name}_ci_upper']:.4f}]"
        )

    # ---------------------------------------------------------
    # 5. Naive baseline
    # ---------------------------------------------------------
    print("\n[5/7] Naive baseline...")

    baseline = naive_baseline(
        y_test,
        train_df[config.data.target_column].value_counts().idxmax(),
    )

    print(f"Baseline accuracy: {baseline['accuracy']:.4f}")

    print(f"Baseline Macro F1: {baseline['macro_f1']:.4f}")

    # ---------------------------------------------------------
    # 6. Save model
    # ---------------------------------------------------------
    print("\n[6/7] Saving model...")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    model_hash = sha256_file(MODEL_PATH)

    print(f"Model: {MODEL_PATH}")
    print(f"SHA-256: {model_hash}")

    # ---------------------------------------------------------
    # 7. MLflow
    # ---------------------------------------------------------
    print("\n[7/7] Logging to MLflow...")

    tracking_path = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")

    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="full_mode") as run:
        mlflow.log_params(
            {
                "mode": "full",
                "vectorizer": "hashing",
                "model": "SGDClassifier",
                "best_C": best_c,
                "max_iter": config.model.max_iter,
                "chunk_size": config.chunking.chunk_size,
                "random_state": config.random_state,
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "rare_class_recall": metrics["rare_class_recall"],
                "accuracy_ci_lower": metrics["accuracy_ci_lower"],
                "accuracy_ci_upper": metrics["accuracy_ci_upper"],
                "macro_f1_ci_lower": metrics["macro_f1_ci_lower"],
                "macro_f1_ci_upper": metrics["macro_f1_ci_upper"],
                "rare_class_recall_ci_lower": metrics["rare_class_recall_ci_lower"],
                "rare_class_recall_ci_upper": metrics["rare_class_recall_ci_upper"],
                "training_time_sec": training_time,
                "baseline_accuracy": baseline["accuracy"],
            }
        )

        mlflow.log_artifact(str(MODEL_PATH))

        run_id = run.info.run_id

    # ---------------------------------------------------------
    # Save results for later labs
    # ---------------------------------------------------------
    results_path = PROJECT_ROOT / "reports" / "LAB2" / "full_results.json"

    results = {
        "mode": "full",
        "run_id": run_id,
        "best_C": best_c,
        "training_time_sec": training_time,
        "model_sha256": model_hash,
        "metrics": metrics,
        "baseline": baseline,
    }

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FULL MODE COMPLETE")
    print("=" * 60)
    print(f"MLflow Run ID: {run_id}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
