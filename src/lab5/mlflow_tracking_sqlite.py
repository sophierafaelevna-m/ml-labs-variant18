"""Регистрация экспериментов ЛР5 в MLflow."""

from pathlib import Path

import mlflow
import pandas as pd

from src.lab5.config import load_config

ROOT = Path("reports/LAB5")


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("LAB5_20Newsgroups_variant18")

    config = load_config("configs/lab5.yaml")
    metrics = pd.read_csv(ROOT / "synthetic_metrics.csv")
    summary = pd.read_csv(ROOT / "comparison_summary.csv")

    for mode in ["similar", "random"]:
        mode_metrics = metrics[metrics["mode"] == mode]
        result = summary[summary["mode"] == mode].iloc[0]

        with mlflow.start_run(run_name=f"lab5_{mode}_seed_{config.seed}") as run:
            mlflow.log_params(
                {
                    "mode": mode,
                    "seed": config.seed,
                    "n_samples": config.n_samples,
                    "dataset": "20 Newsgroups",
                    "variant": 18,
                    "length_multiplier": config.drift.length_multiplier,
                    "label_flip_rate": config.drift.label_flip_rate,
                    "seasonal_amplitude": config.drift.seasonal_amplitude,
                    "seasonal_period": config.drift.seasonal_period,
                }
            )

            mlflow.log_metrics(
                {
                    "passed_features": int(result["passed_features"]),
                    "total_features": int(result["total_features"]),
                    "passed_share": float(result["passed_share"]),
                    "required_share": float(result["required_share"]),
                }
            )

            for _, row in mode_metrics.iterrows():
                feature = row["feature"]

                mlflow.log_metric(
                    f"psi_{feature}",
                    float(row["psi"]),
                )

                if pd.notna(row["ks"]):
                    mlflow.log_metric(
                        f"ks_{feature}",
                        float(row["ks"]),
                    )

            mlflow.set_tag(
                "verdict",
                str(result["verdict"]),
            )

            for filename in [
                "synthetic_metrics.csv",
                "comparison_summary.csv",
                "drift_experiments.csv",
                "generator_config.json",
                "lab5_report.md",
            ]:
                mlflow.log_artifact(str(ROOT / filename))

            print(f"{mode}: {result['verdict']}, run_id={run.info.run_id}")

    print("\nЭксперименты сохранены в SQLite")


if __name__ == "__main__":
    main()
