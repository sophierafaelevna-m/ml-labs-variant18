"""Validated configuration for Lab 3."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    train_path: Path
    test_path: Path
    text_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    validation_fraction: float = Field(gt=0, lt=0.5)


class ModelConfig(StrictModel):
    architecture: Literal["embeddingbag_mlp"]
    vocab_size: int = Field(ge=100)
    min_token_frequency: int = Field(ge=1)
    embedding_dim: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainingConfig(StrictModel):
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)
    seed: int = Field(ge=0)
    device: Literal["auto", "cpu", "mps"]


class LoadingConfig(StrictModel):
    modes: list[Literal["full", "chunk"]] = Field(min_length=2, max_length=2)
    chunk_size: int = Field(gt=0)


class MlflowConfig(StrictModel):
    experiment_name: str = Field(min_length=1)


class Lab3Config(StrictModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loading: LoadingConfig
    mlflow: MlflowConfig


def load_config(path: str | Path = "configs/lab3.yaml") -> Lab3Config:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    config = Lab3Config.model_validate(raw)
    if set(config.loading.modes) != {"full", "chunk"}:
        raise ValueError("loading.modes must contain full and chunk exactly once")
    return config
