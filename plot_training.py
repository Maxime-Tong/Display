from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training history is empty: {path}")
    return {name: [float(row[name]) for row in rows] for name in rows[0]}


def plot_history(history_path, output_dir, log_scale=False):
    history = load_history(history_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for name in ("total", "power_term", "metam_term", "weber_term", "ssim_term"):
        plt.plot(history["step"], history[name], label=name)
    if log_scale:
        plt.yscale("symlog", linthresh=1e-8)
    plt.xlabel("Training step")
    plt.ylabel("Weighted loss contribution" if not log_scale else "Weighted loss contribution (symlog)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "losses.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(history["step"], [100 * value for value in history["power_saving"]])
    plt.xlabel("Training step")
    plt.ylabel("Weighted power saving (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "power_saving.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot screen-adaptor training history")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--log-scale", action="store_true")
    args = parser.parse_args()
    plot_history(args.history, args.output_dir, args.log_scale)


if __name__ == "__main__":
    main()
