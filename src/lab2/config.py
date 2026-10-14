from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: Path
    test_path: Path
    text_column: str
    target_column: str


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vectorizer: str
    model: str


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    C: float = Field(gt=0)
    max_iter: int = Field(gt=0)


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str


class Lab2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    pipeline: PipelineConfig
    chunking: ChunkConfig
    model: ModelConfig
    random_state: int
    mlflow: MlflowConfig


def load_config(path: str | Path) -> Lab2Config:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    return Lab2Config.model_validate(raw_config)
