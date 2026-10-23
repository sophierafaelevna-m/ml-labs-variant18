"""Pydantic configuration for Lab 4, variant 18."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    lab3_config_path: Path
    checkpoint_path: Path


class FeatureConfig(StrictModel):
    length_transform: Literal["log1p"]
    frequency_bins: int = Field(ge=2, le=20)
    use_oov_ratio: bool


class DistanceConfig(StrictModel):
    method: Literal["standardized_euclidean"]
    epsilon: float = Field(gt=0)
    threshold: float = Field(gt=0)


class DecisionConfig(StrictModel):
    confidence_threshold: float = Field(gt=0, lt=1)
    min_foreign_recall: float = Field(gt=0, le=1)
    max_false_reject_rate: float = Field(ge=0, lt=1)


class LoadingConfig(StrictModel):
    modes: list[Literal["full", "chunk"]]
    chunk_size: int = Field(gt=0)

    @model_validator(mode="after")
    def check_modes(self):
        if len(self.modes) != 2 or set(self.modes) != {"full", "chunk"}:
            raise ValueError("Specify full and chunk exactly once")
        return self


class MlflowConfig(StrictModel):
    experiment_name: str = Field(min_length=1)


class Lab4Config(StrictModel):
    source: SourceConfig
    features: FeatureConfig
    distance: DistanceConfig
    decision: DecisionConfig
    loading: LoadingConfig
    mlflow: MlflowConfig


def load_config(path: str | Path = "configs/lab4.yaml") -> Lab4Config:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return Lab4Config.model_validate(raw)


if __name__ == "__main__":
    config = load_config()
    print("PASS: Lab 4 configuration is valid")
    print(config.model_dump_json(indent=2))

    for name, path in {
        "Lab 3 configuration": config.source.lab3_config_path,
        "Lab 3 checkpoint": config.source.checkpoint_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
        print(f"PASS: {name} exists")
