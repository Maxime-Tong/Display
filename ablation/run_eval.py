from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _common import evaluate, write_result


def report(results_dir):
    rows = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" in data and "config" in data:
            rows.append({"name": str(path.relative_to(results_dir).with_suffix("")), **data["config"], **data["summary"]})
    if not rows:
        raise ValueError("no ablation result JSON files found")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (Path(results_dir) / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    (Path(results_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"reported {len(rows)} runs")


def main():
    parser = argparse.ArgumentParser(description="Evaluate one screen-adaptor ablation")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--results-dir")
    parser.add_argument("--model-dir"); parser.add_argument("--data-dir"); parser.add_argument("--out")
    parser.add_argument("--mode", choices=("single", "cluster"), default="cluster")
    parser.add_argument("--repr", choices=("lut", "net"), default="lut")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--lut-resolution", type=int)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--match-distance-threshold", type=float, default=1.8)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.report:
        return report(args.results_dir or args.out)
    if not all((args.model_dir, args.data_dir, args.out)):
        parser.error("--model-dir, --data-dir and --out are required for evaluation")
    if args.block_size < 1 or args.max_images < 0:
        parser.error("block size must be positive and max-images non-negative")
    config = {"mode": args.mode, "representation": args.repr, "block_size": args.block_size, "lut_resolution": args.lut_resolution}
    rows, summary = evaluate(args.model_dir, args.data_dir, args.mode, args.repr, args.block_size, args.max_images, args.device, args.lut_resolution, args.match_distance_threshold)
    write_result(args.out, config, rows, summary)
    print("summary", summary)


if __name__ == "__main__":
    main()
