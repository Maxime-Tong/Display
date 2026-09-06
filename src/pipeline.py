from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from .losses import CombinedLoss, LossConfig
from .model import ColorModel, generate_lut, load_model, save_model
from .power import power_saving
from .scenes import IMAGE_EXTENSIONS, cluster_dkl_scenes, extract_dkl_feature, load_image, load_scene_manifest, match_scene, save_scene_manifest


def image_paths(data_dir):
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def sample_images_per_scene(data_dir, samples_per_scene, seed=0):
    """Sample without replacement from each immediate scene directory."""
    root = Path(data_dir)
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    if not scenes:
        raise FileNotFoundError(f"no scene sub-directories under {root}")
    rng = random.Random(seed)
    selected = []
    for scene in scenes:
        paths = image_paths(scene)
        sampled = rng.sample(paths, min(samples_per_scene, len(paths)))
        selected.extend(sampled)
        print(f"{scene.name}: sampled {len(sampled)} of {len(paths)} images")
    if not selected:
        raise ValueError(f"no images found under {root}")
    return selected


def validate_config(config):
    """Validate the project config and return its loss settings."""
    allowed = {"_comment", "image_size", "batch_size", "steps", "lr", "device", "hidden_dim", "depth", "lut_resolution", "clusters", "log_interval", "seed", "pretrain_samples_per_scene", "loss"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown config items: {', '.join(sorted(unknown))}")
    positive_ints = ("image_size", "batch_size", "steps", "hidden_dim", "depth", "lut_resolution", "log_interval")
    for name in positive_ints:
        if name in config and (not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if "pretrain_samples_per_scene" in config and (not isinstance(config["pretrain_samples_per_scene"], int) or isinstance(config["pretrain_samples_per_scene"], bool) or config["pretrain_samples_per_scene"] <= 0):
        raise ValueError("pretrain_samples_per_scene must be a positive integer")
    if not isinstance(config.get("lr", 1e-3), (int, float)) or config.get("lr", 1e-3) <= 0:
        raise ValueError("lr must be positive")
    if config.get("clusters", 0) < 0:
        raise ValueError("clusters must be non-negative")
    device = config.get("device", "cpu")
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
        raise ValueError("device must be 'cpu' or a CUDA device")

    if "loss" not in config or not isinstance(config["loss"], dict):
        raise ValueError("config must contain a loss object")
    unknown_loss = set(config["loss"]) - set(LossConfig.__dataclass_fields__)
    if unknown_loss:
        raise ValueError(f"unknown loss config items: {', '.join(sorted(unknown_loss))}")
    active_loss = LossConfig(**config["loss"])
    if not 0 < active_loss.target_alpha <= 1:
        raise ValueError("target_alpha must be in (0, 1]")
    if any(value < 0 for value in (active_loss.lambda_power, active_loss.lambda_metam, active_loss.lambda_weber, active_loss.lambda_ssim)):
        raise ValueError("loss weights must be non-negative")
    if active_loss.weber_epsilon <= 0:
        raise ValueError("weber_epsilon must be positive")
    if len(active_loss.power_weights) != 3 or any(value < 0 for value in active_loss.power_weights):
        raise ValueError("power_weights must contain three non-negative values")
    return active_loss


def train_model(paths, config, initial_model=None, history_path=None):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = initial_model or ColorModel(config.get("hidden_dim", 32), config.get("depth", 2))
    model = model.to(device).train()
    active_loss = validate_config(config)
    criterion = CombinedLoss(active_loss, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    size, steps = config.get("image_size", 128), config.get("steps", 1000)
    batch_size = config.get("batch_size", 1)
    if not paths:
        raise ValueError("training dataset contains no images")
    rng = np.random.default_rng(seed)
    history = []
    if history_path:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        optimizer.zero_grad()
        batch = [torch.from_numpy(load_image(paths[int(rng.integers(len(paths)))], size)).to(device) for _ in range(batch_size)]
        outputs = [model(image) for image in batch]
        batch_losses = [criterion(image, optimized) for image, optimized in zip(batch, outputs)]
        losses = {name: torch.stack([item[name] for item in batch_losses]).mean() for name in batch_losses[0]}
        losses["total"].backward()
        optimizer.step()
        if step % config.get("log_interval", 50) == 0 or step + 1 == steps:
            saving = float(np.mean([float(power_saving(image.detach(), optimized.detach(), active_loss.power_weights)) for image, optimized in zip(batch, outputs)]))
            values = " ".join(f"{k}={float(v):.6f}" for k, v in losses.items())
            values += f" power_saving={saving:.6f}"
            print(f"step={step + 1}/{steps} {values}")
            if history_path:
                history.append({
                    "step": step + 1,
                    **{key: float(value) for key, value in losses.items()},
                    "power_term": active_loss.lambda_power * float(losses["power"]),
                    "metam_term": active_loss.lambda_metam * float(losses["metam"]),
                    "weber_term": active_loss.lambda_weber * float(losses["weber"]),
                    "ssim_term": active_loss.lambda_ssim * float(losses["ssim"]),
                    "power_saving": saving,
                })
    if history_path:
        with history_path.open("w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=history[0])
            writer.writeheader()
            writer.writerows(history)
    return model.eval()


def pretrain(data_dir, output_dir, config):
    validate_config(config)
    paths = sample_images_per_scene(data_dir, config["pretrain_samples_per_scene"], config.get("seed", 0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = train_model(paths, config, history_path=output / "training_history.csv")
    save_model(model, output / "base_checkpoint.pt")
    torch.save({"lut": generate_lut(model, config.get("lut_resolution", 16), config.get("device", "cpu"))}, output / "base_lut.pt")


def cluster(data_dir, manifest_path, config):
    validate_config(config)
    paths = image_paths(data_dir)
    if not paths:
        raise ValueError("clustering dataset contains no images")
    features = np.stack([extract_dkl_feature(load_image(path, config["image_size"])) for path in paths])
    _, centers, mean, std = cluster_dkl_scenes(features, config["clusters"], config.get("seed", 0))
    lut_paths = [f"cluster_{index}_lut.pt" for index in range(config["clusters"])]
    save_scene_manifest(manifest_path, centers, mean, std, lut_paths)
    print(f"saved {len(centers)} DKL clusters to {manifest_path}")


def finetune(data_dir, output_dir, base_checkpoint, manifest_path, config):
    validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_scene_manifest(manifest_path)
    groups = [[] for _ in manifest["centers"]]
    for path in image_paths(data_dir):
        groups[match_scene(load_image(path, config["image_size"]), manifest)].append(path)
    if len(groups) != config["clusters"]:
        raise ValueError("config clusters does not match the scene manifest")
    lut_paths = []
    for index, paths in enumerate(groups):
        if not paths:
            raise ValueError(f"cluster {index} has no images")
        model = load_model(base_checkpoint)
        if model.hidden_dim != config["hidden_dim"] or model.depth != config["depth"]:
            raise ValueError("finetune model dimensions do not match the base checkpoint")
        model = train_model(paths, config, model, output / f"cluster_{index}_training_history.csv")
        save_model(model, output / f"cluster_{index}_checkpoint.pt")
        lut_path = output / f"cluster_{index}_lut.pt"
        torch.save({"lut": generate_lut(model, config["lut_resolution"], config["device"])}, lut_path)
        lut_paths.append(lut_path.name)
    save_scene_manifest(manifest_path, manifest["centers"], manifest["mean"], manifest["std"], lut_paths)


def main():
    parser = argparse.ArgumentParser(description="Screen-adaptor training phases")
    commands = parser.add_subparsers(dest="command", required=True)
    pretrain_parser = commands.add_parser("pretrain")
    pretrain_parser.add_argument("--data-dir", required=True)
    pretrain_parser.add_argument("--output-dir", default="outputs")
    pretrain_parser.add_argument("--config", default="configs/pretrain_config.json")
    cluster_parser = commands.add_parser("cluster")
    cluster_parser.add_argument("--data-dir", required=True)
    cluster_parser.add_argument("--manifest", default="outputs/scene_manifest.json")
    cluster_parser.add_argument("--config", default="configs/finetune_config.json")
    finetune_parser = commands.add_parser("finetune")
    finetune_parser.add_argument("--data-dir", required=True)
    finetune_parser.add_argument("--output-dir", default="outputs")
    finetune_parser.add_argument("--base-checkpoint", default="outputs/base_checkpoint.pt")
    finetune_parser.add_argument("--manifest", default="outputs/scene_manifest.json")
    finetune_parser.add_argument("--config", default="configs/finetune_config.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.command == "pretrain":
        pretrain(args.data_dir, args.output_dir, config)
    elif args.command == "cluster":
        cluster(args.data_dir, args.manifest, config)
    else:
        finetune(args.data_dir, args.output_dir, args.base_checkpoint, args.manifest, config)


if __name__ == "__main__":
    main()
