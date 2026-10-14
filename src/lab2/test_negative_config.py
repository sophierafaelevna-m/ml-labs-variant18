import copy
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.lab2.config import Lab2Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "lab2.yaml"
LOG = ROOT / "reports" / "LAB2" / "config_negative_test.log"


def main():
    original = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cases = []

    bad_chunk = copy.deepcopy(original)
    bad_chunk["chunking"]["chunk_size"] = 0
    cases.append(("chunk_size=0", bad_chunk))

    bad_c = copy.deepcopy(original)
    bad_c["model"]["C"] = -1
    cases.append(("C=-1", bad_c))

    extra_field = copy.deepcopy(original)
    extra_field["unexpected_option"] = True
    cases.append(("unexpected field", extra_field))

    lines = []
    for name, candidate in cases:
        try:
            Lab2Config.model_validate(candidate)
        except ValidationError as exc:
            lines.append(f"PASS: {name} rejected")
            lines.append(str(exc))
        else:
            lines.append(f"FAIL: {name} was accepted")

    LOG.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(LOG.read_text(encoding="utf-8"))

    if any(line.startswith("FAIL:") for line in lines):
        raise SystemExit("At least one invalid configuration was accepted.")


if __name__ == "__main__":
    main()
