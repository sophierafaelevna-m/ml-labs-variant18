import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"
OUT.mkdir(parents=True, exist_ok=True)

log_path = OUT / "full_run.log"

with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
        [sys.executable, "-m", "src.lab2.train"],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    monitored = psutil.Process(process.pid)
    peak = 0

    while process.poll() is None:
        try:
            peak = max(peak, monitored.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.005)

    return_code = process.wait()

if return_code != 0:
    print(log_path.read_text(encoding="utf-8"))
    raise SystemExit(f"Full training failed: exit code {return_code}")

subprocess.run(
    [sys.executable, "-m", "src.lab2.save_full_results"],
    cwd=ROOT,
    check=True,
)

result_path = OUT / "full_results.json"
result = json.loads(result_path.read_text(encoding="utf-8"))
result["peak_memory_mb"] = peak / (1024 * 1024)
result["memory_measurement"] = "External RSS sampling of training process"

result_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("FULL MODE COMPLETE")
print(f"Peak process RSS: {result['peak_memory_mb']:.1f} MB")
print(f"Training time: {result['training_time_sec']:.3f} sec")
print(f"MLflow run: {result['mlflow_run_id']}")
print("Full results updated")
