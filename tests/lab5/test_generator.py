"""Автоматические тесты генератора ЛР5."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.lab5 import compare, experiments
from src.lab5.config import GeneratorConfig, load_config
from src.lab5.generator import (
    generate,
    load_source,
    save_dataset,
)


@pytest.fixture
def config(tmp_path):
    """Небольшой контролируемый датасет для тестов."""
    classes = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]

    vocabulary = [
        "galaxy orbit rocket planet",
        "belief reason debate opinion",
        "image pixel graphic render",
        "science telescope stellar space",
    ]

    rows = []
    for index, class_name in enumerate(classes):
        rows.append(
            {
                "text": " ".join(vocabulary[index].split() * 10),
                "target": index,
                "target_name": class_name,
            }
        )

    source_path = tmp_path / "train.jsonl"
    pd.DataFrame(rows).to_json(
        source_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    data = load_config("configs/lab5.yaml").model_dump(mode="json")
    data["source"]["train_path"] = str(source_path)
    data["n_samples"] = 120

    return GeneratorConfig.model_validate(data)


def updated(config, **changes):
    """Изменяем настройки с повторной валидацией."""
    data = config.model_dump(mode="json")

    for key, value in changes.items():
        if key == "drift":
            data["drift"].update(value)
        else:
            data[key] = value

    return GeneratorConfig.model_validate(data)


def test_valid_config(config):
    assert config.mode == "similar"
    assert config.seed == 42
    assert len(config.classes) == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"n_samples": -1},
        {"seed": -5},
        {"mode": "unknown"},
        {"drift": {"label_flip_rate": 1.5}},
        {"drift": {"length_multiplier": -1}},
        {"drift": {"seasonal_amplitude": 1.5}},
        {"drift": {"seasonal_period": 0}},
        {
            "drift": {
                "class_probabilities": {
                    "alt.atheism": 0.5,
                    "talk.religion.misc": 0.5,
                }
            }
        },
    ],
)
def test_invalid_config_rejected(config, changes):
    with pytest.raises(ValidationError):
        updated(config, **changes)


def test_duplicate_classes_rejected(config):
    with pytest.raises(ValidationError):
        updated(
            config,
            classes=["alt.atheism", "alt.atheism"],
        )


def test_similar_generation(config):
    data = generate(config)

    assert len(data) == 120
    assert data["text"].str.strip().ne("").all()
    assert set(data["target_name"]).issubset(config.classes)
    assert (data["mode"] == "similar").all()
    assert not data["label_flipped"].any()


def test_random_generation(config):
    data = generate(config, mode="random")

    assert len(data) == 120
    assert (data["mode"] == "random").all()
    assert data["text"].str.split().str.len().between(10, 60).all()


def test_modes_produce_different_texts(config):
    similar = generate(config, mode="similar")
    random = generate(config, mode="random")

    assert similar["text"].tolist() != random["text"].tolist()


def test_unknown_mode_rejected(config):
    with pytest.raises(ValueError):
        generate(config, mode="invalid")


def test_seed_reproducibility(config, tmp_path):
    first = generate(config)
    second = generate(config)

    path1 = tmp_path / "first.csv"
    path2 = tmp_path / "second.csv"

    hash1 = save_dataset(first, path1)
    hash2 = save_dataset(second, path2)

    assert hash1 == hash2
    assert path1.read_bytes() == path2.read_bytes()

    actual_hash = hashlib.sha256(path1.read_bytes()).hexdigest()
    assert hash1 == actual_hash


def test_different_seeds(config):
    first = generate(config)
    second = generate(updated(config, seed=123))

    assert first["target_name"].tolist() != (second["target_name"].tolist())


def test_length_drift(config):
    baseline = generate(config)

    drifted = generate(
        updated(
            config,
            drift={"length_multiplier": 1.5},
        )
    )

    base_mean = baseline["text"].str.split().str.len().mean()
    drift_mean = drifted["text"].str.split().str.len().mean()

    assert drift_mean > base_mean * 1.4


def test_class_drift(config):
    drifted_config = updated(
        config,
        n_samples=600,
        drift={
            "class_probabilities": {
                "alt.atheism": 0.7,
                "talk.religion.misc": 0.1,
                "comp.graphics": 0.1,
                "sci.space": 0.1,
            }
        },
    )

    data = generate(drifted_config)

    share = (data["target_name"] == "alt.atheism").mean()

    assert 0.62 <= share <= 0.78


def test_label_flip_drift(config):
    baseline = generate(config)

    drifted = generate(
        updated(
            config,
            drift={"label_flip_rate": 0.2},
        )
    )

    assert int(drifted["label_flipped"].sum()) == 24

    changed = baseline["target_name"] != drifted["target_name"]

    assert int(changed.sum()) == 24
    assert baseline["text"].tolist() == drifted["text"].tolist()


def test_seasonal_drift(config):
    seasonal_config = updated(
        config,
        n_samples=400,
        drift={
            "seasonal_amplitude": 0.7,
            "seasonal_period": 40,
        },
    )

    data = generate(seasonal_config)
    lengths = data["text"].str.split().str.len().to_numpy()

    phases = np.sin(2 * np.pi * np.arange(len(data)) / 40)

    high = lengths[phases > 0.7].mean()
    low = lengths[phases < -0.7].mean()

    assert high > low * 2


def test_missing_source_rejected(config):
    broken = updated(
        config,
        source={"train_path": "missing_file.jsonl"},
    )

    with pytest.raises(FileNotFoundError):
        generate(broken)


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"

    pd.DataFrame([{"text": "hello", "target": 1}]).to_json(
        path, orient="records", lines=True
    )

    with pytest.raises(ValueError):
        load_source(path)


def test_empty_text_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"

    pd.DataFrame(
        [
            {
                "text": "",
                "target": 0,
                "target_name": "alt.atheism",
            }
        ]
    ).to_json(path, orient="records", lines=True)

    with pytest.raises(ValueError):
        load_source(path)


def test_psi_detects_distribution_change():
    reference = np.arange(100, dtype=float)

    identical = compare.calculate_psi(reference, reference)
    different = compare.calculate_psi(reference, reference + 1000)

    assert identical == pytest.approx(0.0)
    assert different > 0.2


def test_categorical_psi():
    reference = pd.Series(["A"] * 50 + ["B"] * 50)
    different = pd.Series(["A"] * 90 + ["B"] * 10)

    assert compare.categorical_psi(reference, reference, ["A", "B"]) == pytest.approx(
        0.0
    )

    assert compare.categorical_psi(reference, different, ["A", "B"]) > 0.2


def test_comparison_creates_metrics(config, tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "OUTPUT", tmp_path)
    monkeypatch.setattr(compare, "load_config", lambda _: config)

    compare.main()

    metrics = pd.read_csv(tmp_path / "synthetic_metrics.csv")
    summary = pd.read_csv(tmp_path / "comparison_summary.csv")

    assert len(metrics) == 16
    assert set(summary["mode"]) == {"similar", "random"}


def test_experiments_create_artifacts(config, tmp_path, monkeypatch):
    large_config = updated(config, n_samples=1000)

    monkeypatch.setattr(experiments, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(experiments, "load_config", lambda _: large_config)

    experiments.main()

    assert (tmp_path / "drift_experiments.csv").exists()
    assert (tmp_path / "generator_config.json").exists()

    configs = json.loads((tmp_path / "generator_config.json").read_text())

    assert len(configs) == 5
