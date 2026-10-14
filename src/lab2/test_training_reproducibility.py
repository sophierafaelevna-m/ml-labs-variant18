import json
from pathlib import Path

import numpy as np

from src.lab2.config import load_config
from src.lab2.data import load_jsonl
from src.lab2.model import train_final_model
from src.lab2.chunk_mode import train_streaming

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def main():
    config = load_config(ROOT / "configs/lab2.yaml")
    full = json.loads((OUT / "full_results.json").read_text())
    c_value = float(full["best_C"])

    train = load_jsonl(ROOT / config.data.train_path)
    test = load_jsonl(ROOT / config.data.test_path)
    texts = test[config.data.text_column]

    full_a = train_final_model(
        train, config.data.text_column, config.data.target_column, c_value
    )
    full_b = train_final_model(
        train, config.data.text_column, config.data.target_column, c_value
    )
    full_equal = bool(np.array_equal(full_a.predict(texts), full_b.predict(texts)))

    chunk_args = (
        ROOT / config.data.train_path,
        config.data.text_column,
        config.data.target_column,
        config.chunking.chunk_size,
        c_value,
        config.random_state,
    )
    chunk_a = train_streaming(*chunk_args)
    chunk_b = train_streaming(*chunk_args)

    def predict_chunk(result):
        features = result["vectorizer"].transform(texts)
        return result["classifier"].predict(features)

    chunk_equal = bool(np.array_equal(predict_chunk(chunk_a), predict_chunk(chunk_b)))

    result = {
        "test_rows": len(test),
        "seed": config.random_state,
        "full_independent_training_predictions_identical": full_equal,
        "chunk_independent_training_predictions_identical": chunk_equal,
    }

    (OUT / "training_reproducibility.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not (full_equal and chunk_equal):
        raise SystemExit("FAIL: independent training is not reproducible")

    print("PASS: both modes reproducible after independent training")


if __name__ == "__main__":
    main()
