"""Сравнение реальных и синтетических данных для ЛР5."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.lab5.config import load_config
from src.lab5.generator import generate, load_source, save_dataset

OUTPUT = Path("reports/LAB5")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{2,}", str(text).lower())


def build_profile(texts: pd.Series) -> tuple[set, set]:
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    return (
        set(word for word, _ in counts.most_common(100)),
        set(word for word, _ in counts.most_common(1000)),
    )


def extract_features(
    data: pd.DataFrame,
    top100: set,
    top1000: set,
) -> pd.DataFrame:
    rows = []

    for text in data["text"].astype(str):
        tokens = tokenize(text)
        count = len(tokens)

        rows.append(
            {
                "word_count": len(text.split()),
                "char_count": len(text),
                "avg_word_length": (
                    np.mean([len(w) for w in tokens]) if count else 0.0
                ),
                "unique_word_ratio": (len(set(tokens)) / count if count else 0.0),
                "top100_share": (
                    sum(w in top100 for w in tokens) / count if count else 0.0
                ),
                "top1000_share": (
                    sum(w in top1000 for w in tokens) / count if count else 0.0
                ),
                "short_word_share": (
                    sum(len(w) <= 4 for w in tokens) / count if count else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI с границами интервалов по реальным данным."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if not (np.isfinite(reference).all() and np.isfinite(current).all()):
        raise ValueError("PSI: обнаружены нечисловые значения")

    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))

    if len(quantiles) == 1:
        edges = [-np.inf, quantiles[0], np.inf]
    else:
        edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
        edges = np.unique(edges)

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    epsilon = 1e-6

    ref_share = ref_counts / len(reference)
    cur_share = cur_counts / len(current)

    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    categories: list[str],
) -> float:
    ref_share = reference.value_counts(normalize=True)
    cur_share = current.value_counts(normalize=True)

    epsilon = 1e-6
    total = 0.0

    for category in categories:
        p = max(float(ref_share.get(category, 0)), epsilon)
        q = max(float(cur_share.get(category, 0)), epsilon)

        total += (q - p) * np.log(q / p)

    return float(total)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/lab5.yaml")
    real = load_source(config.source.train_path)

    top100, top1000 = build_profile(real["text"])
    real_features = extract_features(real, top100, top1000)

    results = []

    for mode in ("similar", "random"):
        synthetic = generate(config, mode=mode)

        save_dataset(
            synthetic,
            OUTPUT / f"synthetic_{mode}.csv",
        )

        synthetic_features = extract_features(synthetic, top100, top1000)

        for feature in real_features.columns:
            reference = real_features[feature].to_numpy()
            current = synthetic_features[feature].to_numpy()

            psi = calculate_psi(reference, current)
            ks = float(ks_2samp(reference, current).statistic)

            passed = psi <= 0.1 and ks <= 0.1 if mode == "similar" else psi >= 0.2

            results.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "psi": round(psi, 5),
                    "ks": round(ks, 5),
                    "passed": bool(passed),
                }
            )

        class_psi = categorical_psi(
            real["target_name"],
            synthetic["target_name"],
            config.classes,
        )

        results.append(
            {
                "mode": mode,
                "feature": "class_distribution",
                "psi": round(class_psi, 5),
                "ks": np.nan,
                "passed": bool(
                    class_psi <= 0.1 if mode == "similar" else class_psi >= 0.2
                ),
            }
        )

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        OUTPUT / "synthetic_metrics.csv",
        index=False,
    )

    print("\n=== СРАВНЕНИЕ РЕЖИМОВ ===")
    print(metrics[["mode", "feature", "psi", "ks", "passed"]].to_string(index=False))

    summary = []

    for mode in ("similar", "random"):
        subset = metrics[metrics["mode"] == mode]

        passed_count = int(subset["passed"].sum())
        total_count = len(subset)
        share = passed_count / total_count

        threshold = 0.8 if mode == "similar" else 0.5
        verdict = "PASS" if share >= threshold else "FAIL"

        summary.append(
            {
                "mode": mode,
                "passed_features": passed_count,
                "total_features": total_count,
                "passed_share": round(share, 4),
                "required_share": threshold,
                "verdict": verdict,
            }
        )

        print(f"\n{mode}: {passed_count}/{total_count} характеристик — {verdict}")

    pd.DataFrame(summary).to_csv(
        OUTPUT / "comparison_summary.csv",
        index=False,
    )

    print("\nФайлы сохранены в reports/LAB5/")


if __name__ == "__main__":
    main()
