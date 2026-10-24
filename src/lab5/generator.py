"""Генератор синтетических текстовых данных. ЛР5, вариант 18."""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.lab5.config import GeneratorConfig, load_config


def load_source(path: Path) -> pd.DataFrame:
    """Читаем реальные документы из обучающей выборки."""
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    data = pd.read_json(path, lines=True)

    required = {"text", "target", "target_name"}
    if not required.issubset(data.columns):
        raise ValueError(f"В данных отсутствуют поля: {required - set(data.columns)}")

    if data[list(required)].isna().any().any():
        raise ValueError("В исходных данных есть пропущенные значения")

    if data["text"].astype(str).str.strip().eq("").any():
        raise ValueError("В исходных данных есть пустые тексты")

    return data


def build_vocabulary(texts: pd.Series) -> list[str]:
    """Создаём словарь для режима random."""
    counter = Counter()

    for text in texts:
        words = re.findall(r"[a-zA-Z]{2,}", str(text).lower())
        counter.update(words)

    vocabulary = [word for word, _ in counter.most_common(3000)]

    if not vocabulary:
        raise ValueError("Не удалось сформировать словарь")

    return vocabulary


def change_length(
    text: str,
    multiplier: float,
    rng: np.random.Generator,
) -> str:
    """Изменяем длину текста по заданному коэффициенту."""
    if abs(multiplier - 1.0) < 1e-12:
        return text

    words = text.split()

    if not words:
        return text

    target_length = max(
        1,
        min(15000, round(len(words) * multiplier)),
    )

    if target_length <= len(words):
        return " ".join(words[:target_length])

    extra_count = target_length - len(words)
    extra_words = rng.choice(
        words,
        size=extra_count,
        replace=True,
    ).tolist()

    return " ".join(words + extra_words)


def generate(
    config: GeneratorConfig,
    mode: str | None = None,
) -> pd.DataFrame:
    """Создаём синтетическую выборку."""
    mode = mode or config.mode

    if mode not in {"similar", "random"}:
        raise ValueError(f"Неизвестный режим: {mode}")

    rng = np.random.default_rng(config.seed)
    source = load_source(config.source.train_path)

    classes = config.classes

    label_map = {}

    for class_name in classes:
        subset = source[source["target_name"] == class_name]

        if subset.empty:
            raise ValueError(f"В train отсутствует класс: {class_name}")

        targets = subset["target"].unique()

        if len(targets) != 1:
            raise ValueError(f"У класса {class_name} несколько кодов меток")

        label_map[class_name] = int(targets[0])

    texts_by_class = {
        class_name: source.loc[source["target_name"] == class_name, "text"]
        .astype(str)
        .tolist()
        for class_name in classes
    }

    if config.drift.class_probabilities is None:
        frequencies = source["target_name"].value_counts(normalize=True)
        probabilities = np.array(
            [frequencies[name] for name in classes],
            dtype=float,
        )
        probabilities /= probabilities.sum()
    else:
        probabilities = np.array(
            [config.drift.class_probabilities[name] for name in classes],
            dtype=float,
        )

    vocabulary = build_vocabulary(source["text"]) if mode == "random" else None

    records = []

    for i in range(config.n_samples):
        class_name = str(rng.choice(classes, p=probabilities))

        if mode == "similar":
            candidates = texts_by_class[class_name]
            index = int(rng.integers(0, len(candidates)))
            text = candidates[index]
        else:
            word_count = int(rng.integers(10, 61))
            words = rng.choice(
                vocabulary,
                size=word_count,
                replace=True,
            )
            text = " ".join(words.tolist())

        seasonal_factor = 1.0 + (
            config.drift.seasonal_amplitude
            * np.sin(2 * np.pi * i / config.drift.seasonal_period)
        )

        multiplier = config.drift.length_multiplier * seasonal_factor

        text = change_length(text, multiplier, rng)

        records.append(
            {
                "sample_id": i,
                "text": text,
                "target": label_map[class_name],
                "target_name": class_name,
                "mode": mode,
                "label_flipped": False,
            }
        )

    result = pd.DataFrame(records)

    flip_count = round(config.n_samples * config.drift.label_flip_rate)

    if flip_count > 0:
        flip_indices = rng.choice(
            config.n_samples,
            size=flip_count,
            replace=False,
        )

        for index in flip_indices:
            index = int(index)
            old_class = result.at[index, "target_name"]

            alternatives = [name for name in classes if name != old_class]

            new_class = str(rng.choice(alternatives))

            result.at[index, "target_name"] = new_class
            result.at[index, "target"] = label_map[new_class]
            result.at[index, "label_flipped"] = True

    return result


def save_dataset(data: pd.DataFrame, path: Path) -> str:
    """Сохраняем CSV и возвращаем его SHA256."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/lab5.yaml",
    )

    parser.add_argument(
        "--mode",
        choices=["similar", "random"],
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)
    mode = args.mode or config.mode

    data = generate(config, mode=mode)

    output_path = Path(args.output) if args.output else config.output.path

    checksum = save_dataset(data, output_path)

    print("Генерация завершена!")
    print(f"Режим: {mode}")
    print(f"Документов: {len(data)}")
    print(f"Файл: {output_path}")
    print(f"SHA256: {checksum}")
    print()
    print("Распределение классов:")
    print(data["target_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
