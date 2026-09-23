import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "reports" / "LAB1" / "hash_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    all_ok = True

    for item in manifest["files"]:
        path = ROOT / item["path"]

        actual_hash = sha256_file(path)
        expected_hash = item["sha256"]

        if actual_hash == expected_hash:
            print(f"OK: {item['path']} — совпадает")
        else:
            print(f"ERROR: {item['path']} — хеш НЕ совпадает")
            all_ok = False

    if not all_ok:
        raise SystemExit(1)

    print("Итог: все хеши совпадают.")


if __name__ == "__main__":
    main()
