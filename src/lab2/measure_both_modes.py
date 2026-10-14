import json
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import psutil
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"


def run_measured(command, log_name):
    log_path = OUT / log_name
    peak = 0

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", command],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        monitored = psutil.Process(process.pid)

        while process.poll() is None:
            try:
                peak = max(peak, monitored.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.005)

        code = process.wait()

    if code != 0:
        print(log_path.read_text(encoding="utf-8"))
        raise SystemExit(f"FAIL: {command}, exit code {code}")

    if peak == 0:
        raise SystemExit(f"FAIL: memory not measured for {command}")

    return peak / (1024 * 1024)


def update_results(mode, peak_mb):
    path = OUT / f"{mode}_results.json"
    result = json.loads(path.read_text(encoding="utf-8"))

    if mode == "chunk":
        result["training_window_peak_memory_mb"] = result.get("peak_memory_mb")

    result["peak_memory_mb"] = peak_mb
    result["memory_measurement"] = "External process RSS, entire script lifetime"

    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    client.log_metric(
        result["mlflow_run_id"],
        "peak_memory_mb",
        peak_mb,
        step=1,
    )

    print(
        f"{mode.upper()}: peak RSS = {peak_mb:.1f} MB; run = {result['mlflow_run_id']}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Running FULL mode...", flush=True)
    full_peak = run_measured("src.lab2.train", "full_run.log")

    subprocess.run(
        [sys.executable, "-m", "src.lab2.save_full_results"],
        cwd=ROOT,
        check=True,
    )
    update_results("full", full_peak)

    print("Running CHUNK mode...", flush=True)
    chunk_peak = run_measured("src.lab2.chunk_mode", "chunk_run.log")
    update_results("chunk", chunk_peak)

    subprocess.run(
        [sys.executable, "-m", "src.lab2.compare_modes"],
        cwd=ROOT,
        check=True,
    )

    print("MEMORY COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
