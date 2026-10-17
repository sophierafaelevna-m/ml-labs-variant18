"""Check the Lab 3 configuration, dataset and PyTorch installation."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from src.lab3.config import Lab3Config, load_config


def describe_jsonl(path: Path, text_column: str, target_column: str) -> None:
    counts: Counter[int] = Counter()
    sha256 = hashlib.sha256()
    rows = 0
    with path.open("rb") as file:
        for line in file:
            sha256.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item[text_column], str):
                raise TypeError(f"Expected text in {path}, row {rows + 1}")
            counts[int(item[target_column])] += 1
            rows += 1
    if rows == 0:
        raise ValueError(f"Empty dataset: {path}")
    print(f"{path}: rows={rows}, classes={dict(sorted(counts.items()))}")
    print(f"  sha256={sha256.hexdigest()}")


def check_negative_controls(config: Lab3Config) -> None:
    invalid_cases = [
        ("batch_size=0", {"training": {"batch_size": 0}}),
        ("learning_rate=-1", {"training": {"learning_rate": -1}}),
        ("unknown model field", {"model": {"unknown_parameter": 123}}),
    ]
    for name, changes in invalid_cases:
        candidate = config.model_dump(mode="python")
        for section, values in changes.items():
            candidate[section].update(values)
        try:
            Lab3Config.model_validate(candidate)
        except ValidationError:
            print(f"PASS: invalid configuration rejected ({name})")
        else:
            raise AssertionError(f"Invalid configuration accepted: {name}")


def main() -> None:
    config = load_config()
    print("PASS: Lab 3 configuration is valid")
    check_negative_controls(config)
    describe_jsonl(
        config.data.train_path, config.data.text_column, config.data.target_column
    )
    describe_jsonl(
        config.data.test_path, config.data.text_column, config.data.target_column
    )
    try:
        import torch
    except ImportError:
        print("PyTorch not installed: run python -m pip install torch")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
    print("PASS: Lab 3 initial checks finished")


if __name__ == "__main__":
    main()
