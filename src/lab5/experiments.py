"""Эксперименты с управляемым дрейфом. ЛР5, вариант 18."""

import json
from pathlib import Path

import pandas as pd

from src.lab5.config import GeneratorConfig, load_config
from src.lab5.generator import generate, save_dataset

OUTPUT_DIR = Path("reports/LAB5")


def make_config(
    base: GeneratorConfig,
    changes: dict,
) -> GeneratorConfig:
    """Создаём новую валидированную конфигурацию."""
    data = base.model_dump(mode="json")
    data["drift"].update(changes)
    return GeneratorConfig.model_validate(data)


def describe(
    data: pd.DataFrame,
    experiment: str,
    checksum: str,
) -> dict:
    """Рассчитываем характеристики выборки."""
    lengths = data["text"].str.split().str.len()

    result = {
        "experiment": experiment,
        "n_samples": len(data),
        "mean_words": round(float(lengths.mean()), 2),
        "median_words": round(float(lengths.median()), 2),
        "flipped_labels": int(data["label_flipped"].sum()),
        "sha256": checksum,
    }

    for class_name in sorted(data["target_name"].unique()):
        result[f"share_{class_name}"] = round(
            float((data["target_name"] == class_name).mean()),
            4,
        )

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_config("configs/lab5.yaml")

    scenarios = {
        "baseline": {},
        "length_drift": {
            "length_multiplier": 1.5,
        },
        "class_drift": {
            "class_probabilities": {
                "alt.atheism": 0.70,
                "talk.religion.misc": 0.10,
                "comp.graphics": 0.10,
                "sci.space": 0.10,
            },
        },
        "label_drift": {
            "label_flip_rate": 0.20,
        },
        "seasonal_drift": {
            "seasonal_amplitude": 0.70,
            "seasonal_period": 100,
        },
    }

    results = []
    configurations = {}

    for name, changes in scenarios.items():
        config = make_config(base, changes)

        data = generate(config, mode="similar")

        output_path = OUTPUT_DIR / f"synthetic_{name}.csv"
        checksum = save_dataset(data, output_path)

        results.append(describe(data, name, checksum))

        configurations[name] = config.model_dump(mode="json")

        print(f"\nЭксперимент: {name}")
        print(f"Документов: {len(data)}")
        print(f"Средняя длина: {results[-1]['mean_words']} слов")
        print(f"Изменённых меток: {results[-1]['flipped_labels']}")
        print(f"SHA256: {checksum}")

    summary = pd.DataFrame(results)
    summary.to_csv(
        OUTPUT_DIR / "drift_experiments.csv",
        index=False,
    )

    config_path = OUTPUT_DIR / "generator_config.json"
    config_path.write_text(
        json.dumps(
            configurations,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = summary.set_index("experiment")

    base_length = baseline.loc["baseline", "mean_words"]
    drift_length = baseline.loc["length_drift", "mean_words"]

    class_share = baseline.loc["class_drift", "share_alt.atheism"]

    flipped = baseline.loc["label_drift", "flipped_labels"]

    print("\n=== ПРОВЕРКА ДРЕЙФА ===")

    print(
        "Изменение длины:",
        round(drift_length / base_length, 3),
        "раза",
    )

    print(
        "Доля alt.atheism после изменения:",
        round(class_share * 100, 2),
        "%",
    )

    print("Изменённых меток:", flipped)

    assert drift_length > base_length * 1.25, "Дрейф длины не обнаружен"

    assert class_share > 0.60, "Дрейф распределения классов не обнаружен"

    assert flipped == 200, "Количество изменённых меток неверное"

    print("\nPASS: три проверки дрейфа пройдены")
    print("Сезонность проверим отдельным тестом.")
    print("Результаты сохранены в reports/LAB5/")


if __name__ == "__main__":
    main()
