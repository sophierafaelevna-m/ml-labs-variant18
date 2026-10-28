"""Initialize the Lab 6 model registry from Lab 3 artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.lab6.model_loader import calculate_sha256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/LAB6/registry.json"
METRICS_PATH = ROOT / "reports/LAB3/dl_metrics.csv"

MODELS = {
    "v1": {
        "path": "reports/LAB3/dl_model.pt",
        "mode": "full",
        "status": "working",
    },
    "v2": {
        "path": "reports/LAB3/dl_model_chunk.pt",
        "mode": "chunk",
        "status": "canary",
    },
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_metrics(mode: str) -> list[dict]:
    """Read measured metrics for a training mode."""
    with METRICS_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    selected = [row for row in rows if row.get("mode", "").lower() == mode]

    if not selected:
        raise ValueError(f"No metrics for mode={mode} in {METRICS_PATH}")

    return selected


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Registry already exists: {OUTPUT}. Do not overwrite version history."
        )

    now = timestamp()
    versions = {}
    history = []

    for version, metadata in MODELS.items():
        path = ROOT / metadata["path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        versions[version] = {
            "version": version,
            "path": metadata["path"],
            "sha256": calculate_sha256(path),
            "status": metadata["status"],
            "training_mode": metadata["mode"],
            "metrics_source": "reports/LAB3/dl_metrics.csv",
            "metrics": read_metrics(metadata["mode"]),
        }

        history.append(
            {
                "timestamp": now,
                "version": version,
                "from_status": None,
                "to_status": metadata["status"],
                "action": "register",
                "reason": "Initial registration of Lab 3 checkpoint",
            }
        )

    registry = {
        "schema_version": 1,
        "active_version": "v1",
        "versions": versions,
        "history": history,
    }

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)

    print(f"Registry created: {OUTPUT}")
    print("Active version: v1")

    for version, metadata in versions.items():
        print(
            f"{version}: {metadata['status']}, "
            f"metrics={len(metadata['metrics'])}, "
            f"sha256={metadata['sha256'][:16]}..."
        )


if __name__ == "__main__":
    main()
