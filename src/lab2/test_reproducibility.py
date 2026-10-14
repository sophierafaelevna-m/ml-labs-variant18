import json
from pathlib import Path

import joblib
import numpy as np

from src.lab2.data import load_jsonl
from src.lab2.train import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def main():
    test = load_jsonl(ROOT / "data" / "raw" / "test.jsonl")

    # Full-mode artifact is a sklearn Pipeline.
    full_path = OUT / "ml_model.joblib"
    full_model = joblib.load(full_path)
    full_pred_1 = full_model.predict(test["text"])
    full_pred_2 = full_model.predict(test["text"])

    if not np.array_equal(full_pred_1, full_pred_2):
        raise SystemExit("Full model predictions are not reproducible.")

    # Chunk-mode artifact stores vectorizer and classifier separately.
    chunk_path = OUT / "chunk_model.joblib"
    chunk_model = joblib.load(chunk_path)
    X_test = chunk_model["vectorizer"].transform(test["text"])
    chunk_pred_1 = chunk_model["classifier"].predict(X_test)
    chunk_pred_2 = chunk_model["classifier"].predict(X_test)

    if not np.array_equal(chunk_pred_1, chunk_pred_2):
        raise SystemExit("Chunk model predictions are not reproducible.")

    results = {
        "full_model_sha256": sha256_file(full_path),
        "chunk_model_sha256": sha256_file(chunk_path),
        "full_predictions_identical": True,
        "chunk_predictions_identical": True,
        "test_rows": len(test),
    }

    path = OUT / "reproducibility.json"
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
