from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import ColorModel, save_model
from src.pipeline import cluster, finetune, pretrain, validate_config


def load_config(path, clusters=None, lut_resolution=None, seed=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if clusters is not None: config["clusters"] = clusters
    if lut_resolution is not None: config["lut_resolution"] = lut_resolution
    if seed is not None: config["seed"] = seed
    validate_config(config)
    return config


def train(data_dir, output, recipe, clusters, pretrain_config, finetune_config, seed):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pre = load_config(pretrain_config, seed=seed)
    fine = load_config(finetune_config, clusters=clusters, seed=seed)
    if recipe in ("pretrain", "pretrain_finetune") and not (output / "base_checkpoint.pt").exists():
        pretrain(data_dir, output, pre)
    if recipe == "pretrain": return
    if all((output / f"cluster_{i}_checkpoint.pt").exists() for i in range(clusters)):
        print(f"skip completed recipe: {output}")
        return
    manifest = output / "scene_manifest.json"
    cluster(data_dir, manifest, fine)
    if recipe == "finetune":
        base = output / "base_checkpoint.pt"
        save_model(ColorModel(fine["hidden_dim"], fine["depth"]), base)
    finetune(data_dir, output, output / "base_checkpoint.pt", manifest, fine)


def main():
    parser = argparse.ArgumentParser(description="Train ablation recipes")
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recipe", choices=("pretrain", "finetune", "pretrain_finetune"), required=True)
    parser.add_argument("--clusters", type=int, default=4); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-config", default=str(ROOT / "configs/pretrain_config.json"))
    parser.add_argument("--finetune-config")
    args = parser.parse_args()
    fine_config = args.finetune_config or (Path(__file__).parent / "configs/from_scratch.json" if args.recipe == "finetune" else ROOT / "configs/finetune_config.json")
    train(args.data_dir, args.output_dir, args.recipe, args.clusters, args.pretrain_config, fine_config, args.seed)


if __name__ == "__main__":
    main()
