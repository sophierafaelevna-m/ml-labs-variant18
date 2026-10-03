import hashlib
import json
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
REPORT_DIR = ROOT / "reports" / "LAB1"

CATEGORIES = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

UPSTREAM_SHA256 = "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610"  # pragma: allowlist secret


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_subset(subset: str) -> Path:
    dataset = fetch_20newsgroups(
        data_home=CACHE_DIR,
        subset=subset,
        categories=CATEGORIES,
        shuffle=True,
        random_state=42,
        remove=(),
    )

    output_path = RAW_DIR / f"{subset}.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for text, target in zip(dataset.data, dataset.target, strict=True):
            record = {
                "split": subset,
                "target": int(target),
                "target_name": dataset.target_names[int(target)],
                "text": text,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    return output_path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        save_subset("train"),
        save_subset("test"),
    ]

    files = []

    for path in paths:
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "dataset": "20 Newsgroups",
        "algorithm": "sha256",
        "upstream_archive_sha256": UPSTREAM_SHA256,
        "categories": CATEGORIES,
        "random_state": 42,
        "files": files,
    }

    manifest_path = REPORT_DIR / "hash_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("Dataset downloaded.")
    print(f"Manifest saved to: {manifest_path}")

    for item in files:
        print(f"{item['path']}: {item['sha256']}")


if __name__ == "__main__":
    main()
