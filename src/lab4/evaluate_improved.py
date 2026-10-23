"""Evaluate and save the improved Lab 4 validator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.check_real_data import make_feature_batches
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.length_guard import (
    fit_length_limit,
    reject_with_length,
    token_lengths,
)
from src.lab4.model_confidence import (
    load_frozen_model,
    predict_with_confidence,
)
from src.lab4.negative_checks import auc_pairwise, make_negative_examples
from src.lab4.validator import (
    distance_to_training,
    fit_chunk,
    reject_predictions,
)


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()
    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = json.loads(
        (output_dir / "calibration_preview.json").read_text(encoding="utf-8")
    )
    confidence_threshold = float(calibration["selected_confidence_threshold"])
    distance_threshold = float(calibration["distance_threshold"])

    print("1. Loading frozen Lab 3 model...")
    model, vocabulary = load_frozen_model(lab4.source.checkpoint_path)

    print("2. Fitting training statistics...")
    stats = fit_chunk(make_feature_batches(lab3, lab4, vocabulary, "train"))

    train_texts = (
        record[lab3.data.text_column] for record in iter_split_records(lab3, "train")
    )
    length_limit = fit_length_limit(train_texts, quantile=0.99)

    print("3. Preparing evaluation examples...")
    records = list(iter_split_records(lab3, "val"))
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(records))
    evaluation_indices = indices[len(indices) // 2 :]

    own_records = [records[int(i)] for i in evaluation_indices]
    own_texts = [record[lab3.data.text_column] for record in own_records]

    negative_examples = make_negative_examples(own_texts)
    negative_texts = [text for _, text in negative_examples]
    all_texts = own_texts + negative_texts

    groups = ["own"] * len(own_texts) + [group for group, _ in negative_examples]

    print("4. Calculating predictions and rejection scores...")
    features = extract_feature_matrix(
        all_texts,
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )
    distances = distance_to_training(
        features,
        stats,
        epsilon=lab4.distance.epsilon,
    )
    predictions, confidences = predict_with_confidence(
        all_texts,
        model,
        vocabulary,
    )

    baseline_rejected = reject_predictions(
        distances,
        confidences,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
    )
    rejected = reject_with_length(
        baseline_rejected,
        all_texts,
        length_limit,
    )

    lengths = token_lengths(all_texts)
    scores = np.maximum.reduce(
        [
            distances / distance_threshold,
            confidence_threshold / np.maximum(confidences, 1e-12),
            lengths / length_limit,
        ]
    )

    n_own = len(own_texts)
    tp = int(rejected[n_own:].sum())
    fn = len(negative_texts) - tp
    fp = int(rejected[:n_own].sum())
    tn = n_own - fp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_reject_rate = fp / n_own
    roc_auc = auc_pairwise(scores[:n_own], scores[n_own:])

    print("5. Saving validator parameters...")
    params_path = output_dir / "validator_params.npz"
    np.savez(
        params_path,
        class_ids=stats.class_ids,
        counts=stats.counts,
        means=stats.means,
        variances=stats.variances,
        length_limit=np.array(length_limit),
        distance_threshold=np.array(distance_threshold),
        confidence_threshold=np.array(confidence_threshold),
        epsilon=np.array(lab4.distance.epsilon),
    )
    sha256 = hashlib.sha256(params_path.read_bytes()).hexdigest()

    metrics = {
        "experiment": "development_synthetic_ood",
        "own_rows": n_own,
        "foreign_rows": len(negative_texts),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "precision_rejection": precision,
        "recall_rejection": recall,
        "false_reject_rate": false_reject_rate,
        "roc_auc": roc_auc,
        "length_limit": length_limit,
        "distance_threshold": distance_threshold,
        "confidence_threshold": confidence_threshold,
        "params_sha256": sha256,
    }

    metrics_path = output_dir / "validator_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    print("6. Saving negative test log...")
    log_lines = [
        "Lab 4 negative checks (development experiment)",
        "Foreign examples are synthetic OOD proxies.",
        "This is not an independent final test.",
        "",
    ]

    for group in sorted(set(groups) - {"own"}):
        selected = [i for i, name in enumerate(groups) if name == group]
        detected = int(rejected[selected].sum())
        log_lines.append(f"{group}: {detected}/{len(selected)} rejected")

    log_lines.extend(["", "Accepted foreign examples:"])

    for i in range(n_own, len(all_texts)):
        if not rejected[i]:
            log_lines.append(
                f"index={i}, group={groups[i]}, "
                f"distance={distances[i]:.4f}, "
                f"confidence={confidences[i]:.4f}, "
                f"length={lengths[i]}"
            )

    true_labels = np.asarray(
        [int(record[lab3.data.target_column]) for record in own_records[:12]]
    )
    flipped_labels = (true_labels + 1) % len(stats.class_ids)
    flagged_flips = int(np.sum(predictions[:12] != flipped_labels))
    log_lines.extend(
        [
            "",
            f"Flipped-label mismatches: {flagged_flips}/12",
            "This supervised control requires reference labels.",
            "A mismatch does not prove the reference label is wrong.",
        ]
    )

    log_path = output_dir / "validator_negative.log"
    log_path.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("False rejection rate:", round(false_reject_rate, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Foreign accepted:", fn)
    print("Flipped-label mismatches:", flagged_flips, "/ 12")
    print("Parameter SHA256:", sha256)

    if recall >= lab4.decision.min_foreign_recall:
        print("PASS: foreign recall meets threshold")
    else:
        print("WARNING: foreign recall below threshold")

    if false_reject_rate <= lab4.decision.max_false_reject_rate:
        print("PASS: false rejection rate meets threshold")
    else:
        print("WARNING: false rejection rate above threshold")

    print()
    print("Saved:", params_path)
    print("Saved:", metrics_path)
    print("Saved:", log_path)
    print("PASS: improved validator evaluation completed")


if __name__ == "__main__":
    main()
