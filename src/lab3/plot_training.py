"""Plot learning curves after both training modes have finished."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("reports/LAB3")


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in ("full", "chunk"):
        path = ROOT / f"training_history_{mode}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run training for {mode} first: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(
            epochs, [float(row["train_loss"]) for row in rows], label=f"{mode} train"
        )
        ax.plot(
            epochs, [float(row["val_loss"]) for row in rows], "--", label=f"{mode} val"
        )
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Lab 3: learning curves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output = ROOT / "learning_curves.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"PASS: saved {output}")


if __name__ == "__main__":
    main()
