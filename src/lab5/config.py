"""Pydantic-конфигурация генератора для ЛР5."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    train_path: Path


class DriftConfig(StrictModel):
    length_multiplier: float = Field(default=1.0, gt=0)
    class_probabilities: dict[str, float] | None = None
    label_flip_rate: float = Field(default=0.0, ge=0, le=1)
    seasonal_amplitude: float = Field(default=0.0, ge=0, le=0.9)
    seasonal_period: int = Field(default=100, ge=2)


class OutputConfig(StrictModel):
    path: Path
    format: Literal["csv"] = "csv"


class GeneratorConfig(StrictModel):
    mode: Literal["similar", "random"]
    n_samples: int = Field(gt=0)
    seed: int = Field(ge=0)
    source: SourceConfig
    classes: list[str] = Field(min_length=2)
    drift: DriftConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_probabilities(self):
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Названия классов должны быть уникальными")

        probabilities = self.drift.class_probabilities

        if probabilities is not None:
            if set(probabilities) != set(self.classes):
                raise ValueError("Вероятности нужны для всех классов")

            if any(v < 0 or v > 1 for v in probabilities.values()):
                raise ValueError("Вероятности должны быть от 0 до 1")

            if abs(sum(probabilities.values()) - 1.0) > 1e-8:
                raise ValueError("Сумма вероятностей должна быть равна 1")

        return self


def load_config(path: str | Path) -> GeneratorConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return GeneratorConfig.model_validate(data)


if __name__ == "__main__":
    config = load_config("configs/lab5.yaml")
    print("Конфигурация ЛР5 успешно проверена!")
    print(f"Режим: {config.mode}")
    print(f"Объём выборки: {config.n_samples}")
    print(f"Seed: {config.seed}")
    print(f"Количество классов: {len(config.classes)}")
    print(f"Доля переворота меток: {config.drift.label_flip_rate}")
