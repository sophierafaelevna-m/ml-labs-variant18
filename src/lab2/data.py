import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSONL dataset into a DataFrame."""
    rows = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    return pd.DataFrame(rows)


def split_train_validation(
    data: pd.DataFrame,
    target_column: str,
    random_state: int,
    validation_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation sets."""
    train_data, validation_data = train_test_split(
        data,
        test_size=validation_size,
        stratify=data[target_column],
        random_state=random_state,
    )

    return train_data.reset_index(drop=True), validation_data.reset_index(drop=True)
