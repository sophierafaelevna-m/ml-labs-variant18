import hashlib
import json
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import mlflow
import numpy as np
import psutil

from src.lab2.config import load_config
from src.lab2.data import load_jsonl
from src.lab2.metrics import naive_baseline
from src.lab2.model import build_classifier, build_vectorizer
from src.lab2.train import calculate_test_metrics_with_ci

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def read_chunks(path, text_column, target_column, chunk_size):
    texts = []
    labels = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            texts.append(str(record[text_column]))
            labels.append(record[target_column])

            if len(texts) == chunk_size:
                yield texts, np.asarray(labels)
                texts = []
                labels = []

    if texts:
        yield texts, np.asarray(labels)


class PeakMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.running = False
        self.thread = None

    def sample(self):
        while self.running:
            self.peak = max(
                self.peak,
                self.process.memory_info().rss,
            )
            time.sleep(0.01)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(
            target=self.sample,
            daemon=True,
        )
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.running = False
        self.thread.join()
        self.peak = max(
            self.peak,
            self.process.memory_info().rss,
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def train_streaming(
    train_path,
    text_column,
    target_column,
    chunk_size,
    c_value,
    random_state,
):
    vectorizer = build_vectorizer()
    classifier = build_classifier(c_value, random_state)

    classes = set()
    label_counts = Counter()
    with Path(train_path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                label = json.loads(line)[target_column]
                classes.add(label)
                label_counts[label] += 1

    classes = np.asarray(sorted(classes))

    chunks_processed = 0
    train_rows = 0

    memory = PeakMemory()
    start = time.perf_counter()

    with memory:
        for texts, labels in read_chunks(
            train_path,
            text_column,
            target_column,
            chunk_size,
        ):
            X_chunk = vectorizer.transform(texts)

            classifier.partial_fit(
                X_chunk,
                labels,
                classes=classes,
            )

            chunks_processed += 1
            train_rows += len(labels)

            del X_chunk

    training_time = time.perf_counter() - start
    peak_memory_mb = memory.peak / (1024 * 1024)

    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "training_time_sec": training_time,
        "peak_memory_mb": peak_memory_mb,
        "chunks_processed": chunks_processed,
        "train_rows": train_rows,
        "majority_label": label_counts.most_common(1)[0][0],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs" / "lab2.yaml")

    full_results = json.loads((OUT / "full_results.json").read_text(encoding="utf-8"))
    best_c = float(full_results["best_C"])

    result = train_streaming(
        ROOT / config.data.train_path,
        config.data.text_column,
        config.data.target_column,
        config.chunking.chunk_size,
        best_c,
        config.random_state,
    )

    model = {
        "vectorizer": result["vectorizer"],
        "classifier": result["classifier"],
        "text_column": config.data.text_column,
        "target_column": config.data.target_column,
    }

    model_path = OUT / "chunk_model.joblib"
    joblib.dump(model, model_path)
    model_hash = sha256_file(model_path)

    test_df = load_jsonl(ROOT / config.data.test_path)
    X_test = result["vectorizer"].transform(test_df[config.data.text_column])
    predictions = result["classifier"].predict(X_test)

    y_test = test_df[config.data.target_column]
    metrics = calculate_test_metrics_with_ci(y_test, predictions)
    baseline = naive_baseline(y_test, result["majority_label"])

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="chunk_mode") as run:
        mlflow.log_params(
            {
                "mode": "chunk",
                "vectorizer": "HashingVectorizer",
                "model": "SGDClassifier",
                "loss": "hinge",
                "C": best_c,
                "chunk_size": config.chunking.chunk_size,
                "random_state": config.random_state,
                "chunks_processed": result["chunks_processed"],
            }
        )

        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_metrics(
            {
                "training_time_sec": result["training_time_sec"],
                "peak_memory_mb": result["peak_memory_mb"],
                "baseline_accuracy": float(baseline["accuracy"]),
                "baseline_macro_f1": float(baseline["macro_f1"]),
            }
        )
        mlflow.log_artifact(str(model_path))

        output = {
            "mode": "chunk",
            "best_C": best_c,
            "training_time_sec": result["training_time_sec"],
            "peak_memory_mb": result["peak_memory_mb"],
            "chunk_size": config.chunking.chunk_size,
            "chunks_processed": result["chunks_processed"],
            "train_rows": result["train_rows"],
            "test_rows": len(test_df),
            "feature_count": result["vectorizer"].n_features,
            "metrics": metrics,
            "baseline": {
                "accuracy": float(baseline["accuracy"]),
                "macro_f1": float(baseline["macro_f1"]),
            },
            "mlflow_run_id": run.info.run_id,
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": model_hash,
            "note": "True incremental training with partial_fit.",
        }

    (OUT / "chunk_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("CHUNK MODE COMPLETE")
    print(f"Chunks: {result['chunks_processed']}")
    print(f"Training time: {result['training_time_sec']:.3f} sec")
    print(f"Peak RSS: {result['peak_memory_mb']:.1f} MB")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"MLflow run: {output['mlflow_run_id']}")


if __name__ == "__main__":
    main()
