import json
from pathlib import Path
from unittest.mock import patch

from sklearn.pipeline import Pipeline

from src.lab2.config import load_config
from src.lab2.data import load_jsonl, split_train_validation
from src.lab2.metrics import naive_baseline
from src.lab2.model import train_final_model, tune_hyperparameter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def main():
    config = load_config(ROOT / "configs/lab2.yaml")
    train = load_jsonl(ROOT / config.data.train_path)

    train_part, validation = split_train_validation(
        train,
        config.data.target_column,
        config.random_state,
    )

    validation = validation.copy()
    marker = "validation_only_leakage_marker_987654"
    validation[config.data.text_column] = (
        validation[config.data.text_column].astype(str) + " " + marker
    )

    original_fit = Pipeline.fit
    fitted_rows = []

    def guarded_fit(self, X, y=None, **kwargs):
        if any(marker in str(text) for text in X):
            raise AssertionError("LEAKAGE: validation data reached model.fit")
        fitted_rows.append(len(X))
        return original_fit(self, X, y, **kwargs)

    with patch.object(Pipeline, "fit", guarded_fit):
        tuning = tune_hyperparameter(
            train_part,
            validation,
            config.data.text_column,
            config.data.target_column,
            [0.1, 1.0, 10.0],
        )

        train_final_model(
            train,
            config.data.text_column,
            config.data.target_column,
            tuning["best_c"],
        )

    expected_rows = [len(train_part)] * 3 + [len(train)]
    training_isolated = fitted_rows == expected_rows

    baseline = naive_baseline(["B", "B", "A"], "A")
    baseline_uses_training_majority = abs(float(baseline["accuracy"]) - 1 / 3) < 1e-12

    result = {
        "validation_marker_excluded_from_training": training_isolated,
        "fit_row_counts": fitted_rows,
        "expected_fit_row_counts": expected_rows,
        "baseline_uses_training_majority": baseline_uses_training_majority,
        "status": (
            "PASS" if training_isolated and baseline_uses_training_majority else "FAIL"
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_test.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] != "PASS":
        raise SystemExit("FAIL: leakage or baseline test")

    print("PASS: validation isolation and baseline test")


if __name__ == "__main__":
    main()
