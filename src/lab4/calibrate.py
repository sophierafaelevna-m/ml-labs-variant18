"""Calibrate the Lab 4 confidence threshold on validation data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.check_real_data import make_feature_batches
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.model_confidence import (
    load_frozen_model,
    predict_with_confidence,
)
from src.lab4.validator import (
    distance_to_training,
    fit_chunk,
    reject_predictions,
)


def rejection_rate(distances, confidences, distance_threshold, confidence_threshold):
    rejected = reject_predictions(
        distances,
        confidences,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
    )
    return float(np.mean(rejected))


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()

    model, vocabulary = load_frozen_model(lab4.source.checkpoint_path)

    stats = fit_chunk(make_feature_batches(lab3, lab4, vocabulary, "train"))

    records = list(iter_split_records(lab3, "val"))
    texts = [record[lab3.data.text_column] for record in records]

    features = extract_feature_matrix(
        texts,
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )

    distances = distance_to_training(
        features,
        stats,
        epsilon=lab4.distance.epsilon,
    )

    _, confidences = predict_with_confidence(texts, model, vocabulary)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(texts))
    midpoint = len(indices) // 2

    calibration_indices = indices[:midpoint]
    evaluation_indices = indices[midpoint:]

    cal_distances = distances[calibration_indices]
    cal_confidences = confidences[calibration_indices]

    eval_distances = distances[evaluation_indices]
    eval_confidences = confidences[evaluation_indices]

    distance_threshold = lab4.distance.threshold
    initial_confidence_threshold = lab4.decision.confidence_threshold

    target_calibration_frr = 0.08
    candidates = np.linspace(0.01, 0.99, 99)

    valid_thresholds = [
        float(threshold)
        for threshold in candidates
        if rejection_rate(
            cal_distances,
            cal_confidences,
            distance_threshold,
            float(threshold),
        )
        <= target_calibration_frr
    ]

    if not valid_thresholds:
        raise ValueError(
            "Distance threshold alone may reject too many calibration examples"
        )

    selected_threshold = max(valid_thresholds)

    initial_cal_rate = rejection_rate(
        cal_distances,
        cal_confidences,
        distance_threshold,
        initial_confidence_threshold,
    )

    selected_cal_rate = rejection_rate(
        cal_distances,
        cal_confidences,
        distance_threshold,
        selected_threshold,
    )

    initial_eval_rate = rejection_rate(
        eval_distances,
        eval_confidences,
        distance_threshold,
        initial_confidence_threshold,
    )

    selected_eval_rate = rejection_rate(
        eval_distances,
        eval_confidences,
        distance_threshold,
        selected_threshold,
    )

    result = {
        "seed": 42,
        "calibration_rows": len(calibration_indices),
        "evaluation_rows": len(evaluation_indices),
        "distance_threshold": distance_threshold,
        "initial_confidence_threshold": initial_confidence_threshold,
        "selected_confidence_threshold": selected_threshold,
        "target_calibration_false_reject_rate": target_calibration_frr,
        "initial_calibration_false_reject_rate": initial_cal_rate,
        "selected_calibration_false_reject_rate": selected_cal_rate,
        "initial_evaluation_false_reject_rate": initial_eval_rate,
        "selected_evaluation_false_reject_rate": selected_eval_rate,
        "required_max_false_reject_rate": (lab4.decision.max_false_reject_rate),
    }

    output_path = Path("reports/LAB4/calibration_preview.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Calibration rows:", len(calibration_indices))
    print("Evaluation rows:", len(evaluation_indices))
    print("Initial confidence threshold:", initial_confidence_threshold)
    print("Selected confidence threshold:", selected_threshold)

    print("\nFalse rejection rates:")
    print(f"Initial calibration: {initial_cal_rate:.2%}")
    print(f"Selected calibration: {selected_cal_rate:.2%}")
    print(f"Initial evaluation: {initial_eval_rate:.2%}")
    print(f"Selected evaluation: {selected_eval_rate:.2%}")

    if selected_eval_rate <= lab4.decision.max_false_reject_rate:
        print("PASS: evaluation false rejection rate is within 10%")
    else:
        print("WARNING: evaluation false rejection rate exceeds 10%")

    print("\nSaved:", output_path)
    print("PASS: calibration procedure completed")


if __name__ == "__main__":
    main()
