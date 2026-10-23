"""Evaluate Lab 4 rejection on held-out and synthetic foreign texts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.lab3.config import load_config as load_lab3
from src.lab3.text_data import iter_split_records
from src.lab4.check_real_data import make_feature_batches
from src.lab4.config import load_config as load_lab4
from src.lab4.features import extract_feature_matrix
from src.lab4.model_confidence import load_frozen_model, predict_with_confidence
from src.lab4.validator import distance_to_training, fit_chunk, reject_predictions

FOREIGN_TEXTS = [
    "The football goalkeeper saved a penalty during the championship final.",
    "The basketball team won the tournament after a difficult season.",
    "The chef prepared fresh pasta with tomatoes and basil.",
    "This recipe uses butter flour sugar and fresh strawberries.",
    "The patient received a medical examination at the hospital.",
    "Doctors discussed the treatment of a common respiratory infection.",
    "The stock market reacted to the company's quarterly earnings.",
    "Investors reviewed the bank's interest rates and loan portfolio.",
    "The train arrived at the station before the morning rush hour.",
    "The airline announced new flights between international airports.",
    "The gardener planted roses and watered the vegetable beds.",
    "The museum opened an exhibition of paintings and sculptures.",
]


def make_negative_examples(own_texts: list[str]):
    """Create controlled OOD proxies, not real unseen dataset categories."""
    examples = []

    for text in FOREIGN_TEXTS:
        examples.append(("foreign_topic", text))

    for i in range(12):
        unknown = " ".join(f"zzunknown{i}_{j}zz" for j in range(80))
        examples.append(("unknown_vocabulary", unknown))

    for i in range(12):
        examples.append(("length_outlier", own_texts[i] * 25))

    for i in range(12):
        shifted = own_texts[i] + (" zzshiftedunseentoken" * 150)
        examples.append(("distribution_shift", shifted))

    return examples


def auc_pairwise(own_scores: np.ndarray, foreign_scores: np.ndarray) -> float:
    """ROC-AUC: probability that a foreign example has a higher score."""
    differences = foreign_scores[:, None] - own_scores[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def main():
    lab3 = load_lab3()
    lab4 = load_lab4()

    calibration = json.loads(
        Path("reports/LAB4/calibration_preview.json").read_text(encoding="utf-8")
    )
    confidence_threshold = float(calibration["selected_confidence_threshold"])
    distance_threshold = float(calibration["distance_threshold"])

    model, vocabulary = load_frozen_model(lab4.source.checkpoint_path)
    stats = fit_chunk(make_feature_batches(lab3, lab4, vocabulary, "train"))

    records = list(iter_split_records(lab3, "val"))
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(records))
    evaluation_indices = indices[len(indices) // 2 :]

    own_records = [records[int(i)] for i in evaluation_indices]
    own_texts = [record[lab3.data.text_column] for record in own_records]

    negative_examples = make_negative_examples(own_texts)
    negative_texts = [text for _, text in negative_examples]
    all_texts = own_texts + negative_texts

    features = extract_feature_matrix(
        all_texts,
        vocabulary,
        frequency_bins=lab4.features.frequency_bins,
        use_oov_ratio=lab4.features.use_oov_ratio,
    )
    distances = distance_to_training(features, stats, epsilon=lab4.distance.epsilon)
    predictions, confidences = predict_with_confidence(all_texts, model, vocabulary)
    rejected = reject_predictions(
        distances,
        confidences,
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
    )

    # A continuous score is needed for ROC-AUC; larger means more suspicious.
    scores = np.maximum(
        distances / distance_threshold,
        confidence_threshold / np.maximum(confidences, 1e-12),
    )

    n_own = len(own_texts)
    own_rejected = rejected[:n_own]
    foreign_rejected = rejected[n_own:]

    tp = int(foreign_rejected.sum())
    fn = int((~foreign_rejected).sum())
    fp = int(own_rejected.sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_reject_rate = fp / n_own
    roc_auc = auc_pairwise(scores[:n_own], scores[n_own:])

    output_dir = Path("reports/LAB4")
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = ["own"] * n_own + [group for group, _ in negative_examples]
    rows = []

    for i, group in enumerate(groups):
        rows.append(
            {
                "sample_id": i,
                "group": group,
                "expected_reject": int(group != "own"),
                "actual_reject": int(rejected[i]),
                "distance": round(float(distances[i]), 6),
                "confidence": round(float(confidences[i]), 6),
                "rejection_score": round(float(scores[i]), 6),
            }
        )

    # Flipped labels are a separate control: the text itself is unchanged.
    true_labels = np.asarray(
        [int(record[lab3.data.target_column]) for record in own_records]
    )
    flipped_labels = (true_labels[:12] + 1) % 4
    label_mismatches = predictions[:12] != flipped_labels

    for i, mismatch in enumerate(label_mismatches):
        rows.append(
            {
                "sample_id": f"flipped_{i}",
                "group": "flipped_label_separate_control",
                "expected_reject": 1,
                "actual_reject": int(mismatch),
                "distance": round(float(distances[i]), 6),
                "confidence": round(float(confidences[i]), 6),
                "rejection_score": "",
            }
        )

    with (output_dir / "negative_preview.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    group_results = {}
    for group in sorted(set(groups) - {"own"}):
        selected = np.array([i for i, name in enumerate(groups) if name == group])
        group_results[group] = {
            "count": len(selected),
            "rejected": int(rejected[selected].sum()),
        }

    metrics = {
        "own_evaluation_rows": n_own,
        "foreign_examples": len(negative_texts),
        "distance_threshold": distance_threshold,
        "confidence_threshold": confidence_threshold,
        "precision_rejection": precision,
        "recall_rejection": recall,
        "false_reject_rate": false_reject_rate,
        "roc_auc": roc_auc,
        "foreign_accepted": fn,
        "foreign_by_group": group_results,
        "flipped_label_mismatch_count": int(label_mismatches.sum()),
        "flipped_label_control_count": len(label_mismatches),
        "note": (
            "Foreign samples are synthetic OOD proxies. Flipped labels "
            "are evaluated separately, not included in ROC-AUC."
        ),
    }

    (output_dir / "negative_metrics_preview.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Own evaluation rows:", n_own)
    print("Foreign examples:", len(negative_texts))
    print(f"Precision rejection: {precision:.4f}")
    print(f"Recall rejection: {recall:.4f}")
    print(f"False reject rate: {false_reject_rate:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Foreign examples accepted:", fn)
    print("Flipped labels flagged:", int(label_mismatches.sum()), "/ 12")

    for group, result in group_results.items():
        print(f"{group}: {result['rejected']}/{result['count']} rejected")

    if recall >= lab4.decision.min_foreign_recall:
        print("PASS: foreign recall meets the declared threshold")
    else:
        print("WARNING: foreign recall is below the declared threshold")

    if false_reject_rate <= lab4.decision.max_false_reject_rate:
        print("PASS: false rejection rate meets the declared threshold")
    else:
        print("WARNING: false rejection rate exceeds the threshold")

    if fn:
        print("WARNING: some foreign examples were accepted; inspect CSV")

    print("Saved negative_preview.csv and negative_metrics_preview.json")
    print("PASS: negative evaluation completed")


if __name__ == "__main__":
    main()
