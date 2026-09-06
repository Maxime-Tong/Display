from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import ColorModel, generate_lut, save_model
from src.pipeline import cluster, finetune, pretrain, validate_config


def load_config(path, clusters=None, lut_resolution=None, seed=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if clusters is not None: config["clusters"] = clusters
    if lut_resolution is not None: config["lut_resolution"] = lut_resolution
    if seed is not None: config["seed"] = seed
    validate_config(config)
    return config


def train(data_dir, output, recipe, clusters, pretrain_config, finetune_config, seed, max_images=100):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pre = load_config(pretrain_config, seed=seed)
    fine = load_config(finetune_config, clusters=clusters, seed=seed)
    if recipe in ("pretrain", "pretrain_finetune") and not (output / "base_checkpoint.pt").exists():
        pretrain(data_dir, output, pre, max_images)
    if recipe == "pretrain": return
    if all((output / f"cluster_{i}_checkpoint.pt").exists() for i in range(clusters)):
        print(f"skip completed recipe: {output}")
        return
    manifest = output / "scene_manifest.json"
    cluster(data_dir, manifest, fine, max_images)
    if recipe == "finetune":
        base = output / "base_checkpoint.pt"
        base_model = ColorModel(fine["hidden_dim"], fine["depth"])
        save_model(base_model, base)
        torch.save({"lut": generate_lut(base_model, fine["lut_resolution"], fine["device"])}, output / "base_lut.pt")
    finetune(data_dir, output, output / "base_checkpoint.pt", manifest, fine, max_images)


def main():
    parser = argparse.ArgumentParser(description="Train ablation recipes")
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recipe", choices=("pretrain", "finetune", "pretrain_finetune"), required=True)
    parser.add_argument("--clusters", type=int, default=4); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--pretrain-config", default=str(ROOT / "configs/pretrain_config.json"))
    parser.add_argument("--finetune-config")
    args = parser.parse_args()
    fine_config = args.finetune_config or (Path(__file__).parent / "configs/from_scratch.json" if args.recipe == "finetune" else ROOT / "configs/finetune_config.json")
    if args.max_images < 0:
        parser.error("max-images must be non-negative")
    train(args.data_dir, args.output_dir, args.recipe, args.clusters, args.pretrain_config, fine_config, args.seed, args.max_images)


if __name__ == "__main__":
    main()
