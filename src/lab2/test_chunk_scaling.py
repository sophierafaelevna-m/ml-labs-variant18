import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from src.lab2.chunk_mode import train_streaming
from src.lab2.config import load_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def worker(path):
    config = load_config(ROOT / "configs/lab2.yaml")
    full = json.loads((OUT / "full_results.json").read_text())

    result = train_streaming(
        path,
        config.data.text_column,
        config.data.target_column,
        config.chunking.chunk_size,
        float(full["best_C"]),
        config.random_state,
    )

    print(
        "SCALING_RESULT="
        + json.dumps(
            {
                "rows": result["train_rows"],
                "chunks": result["chunks_processed"],
            }
        ),
        flush=True,
    )


def measure(path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.lab2.test_chunk_scaling",
            "--worker",
            str(path),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitored = psutil.Process(process.pid)
    peak = 0

    while process.poll() is None:
        try:
            peak = max(peak, monitored.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.005)

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(stdout)
        print(stderr)
        raise SystemExit("FAIL: chunk scaling worker")

    line = next(
        (line for line in stdout.splitlines() if line.startswith("SCALING_RESULT=")),
        None,
    )

    if line is None:
        raise SystemExit("FAIL: scaling result missing")

    data = json.loads(line.split("=", 1)[1])
    data["peak_memory_mb"] = peak / (1024 * 1024)
    return data


def main():
    config = load_config(ROOT / "configs/lab2.yaml")
    source = ROOT / config.data.train_path
    rows = []

    with tempfile.TemporaryDirectory() as directory:
        for multiplier in [1, 2, 4]:
            target = Path(directory) / f"train_x{multiplier}.jsonl"

            with target.open("wb") as destination:
                for _ in range(multiplier):
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, destination)

            result = measure(target)
            result["data_multiplier"] = multiplier
            rows.append(result)

            print(
                f"x{multiplier}: {result['rows']} rows; "
                f"{result['peak_memory_mb']:.1f} MB"
            )

    base = rows[0]["peak_memory_mb"]
    maximum = max(row["peak_memory_mb"] for row in rows)
    growth = maximum - base
    tolerance = max(30.0, base * 0.20)

    passed = growth <= tolerance

    output = {
        "measurements": rows,
        "peak_growth_mb": growth,
        "tolerance_mb": tolerance,
        "bounded_memory_empirical_check": passed,
        "status": "PASS" if passed else "FAIL",
        "note": (
            "Empirical RSS check on 1x, 2x and 4x datasets. "
            "Does not constitute a mathematical proof of bounded memory."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunk_scaling.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))

    if not passed:
        raise SystemExit("FAIL: chunk memory growth exceeds tolerance")

    print("PASS: chunk memory remains approximately bounded")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        main()
