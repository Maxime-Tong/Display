from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .losses import CombinedLoss, LossConfig
from .model import FactorModel, generate_lut, save_model
from .power import power_saving


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_paths(data_dir):
    return sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path, size=128):
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def sample_images_per_scene(data_dir, samples_per_scene, seed=0):
    root = Path(data_dir)
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    pools = scenes or [root]
    rng, selected = random.Random(seed), []
    for scene in pools:
        paths = image_paths(scene)
        selected.extend(rng.sample(paths, min(samples_per_scene, len(paths))))
    if not selected:
        raise ValueError(f"no images found under {root}")
    return selected


def validate_config(config):
    allowed = {"_comment", "image_size", "batch_size", "steps", "lr", "device",
               "hidden_dim", "depth", "lut_resolution", "log_interval", "seed",
               "samples_per_scene", "loss"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown config items: {', '.join(sorted(unknown))}")
    for name in ("image_size", "batch_size", "steps", "hidden_dim", "depth",
                 "lut_resolution", "log_interval", "samples_per_scene"):
        if not isinstance(config.get(name), int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("max_grad_norm", 1.0) <= 0:
        raise ValueError("max_grad_norm must be positive")
    active_loss = LossConfig(**config["loss"])
    if not 0 < active_loss.target_alpha <= 1:
        raise ValueError("target_alpha must be in (0, 1]")
    return active_loss


def train(data_dir, output_dir, config, max_images=0):
    loss_config = validate_config(config)
    device = torch.device(config["device"])
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    paths = sample_images_per_scene(data_dir, config["samples_per_scene"], seed)
    if max_images:
        paths = paths[:max_images]
    model = FactorModel(config["hidden_dim"], config["depth"],
                        loss_config.target_alpha).to(device).train()
    criterion = CombinedLoss(loss_config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    rng, history, skipped = np.random.default_rng(seed), [], 0
    for step in range(config["steps"]):
        batch = [torch.from_numpy(load_image(paths[int(rng.integers(len(paths)))],
                                             config["image_size"])).to(device)
                 for _ in range(config["batch_size"])]
        optimizer.zero_grad()
        outputs = [model(image) for image in batch]
        items = [criterion(image, output) for image, output in zip(batch, outputs)]
        losses = {name: torch.stack([item[name] for item in items]).mean() for name in items[0]}
        if not torch.isfinite(losses["total"]):
            skipped += 1
            continue
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            skipped += 1
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
        optimizer.step()
        if step % config["log_interval"] == 0 or step + 1 == config["steps"]:
            saving = float(np.mean([float(power_saving(a.detach(), b.detach(), loss_config.power_weights))
                                    for a, b in zip(batch, outputs)]))
            row = {"step": step + 1, **{key: float(value) for key, value in losses.items()},
                   "power_saving": saving}
            history.append(row)
            print(" ".join([f"step={step + 1}/{config['steps']}"] +
                           [f"{key}={value:.6f}" for key, value in row.items() if key != "step"]))
    if skipped:
        print(f"skipped_nonfinite_updates={skipped}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_model(model.eval(), output / "factor_checkpoint.pt")
    torch.save({"lut": generate_lut(model, config["lut_resolution"], device)}, output / "factor_lut.pt")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description="Train the scalar display-power factor LUT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train(args.data_dir, args.output_dir, config, args.max_images)


if __name__ == "__main__":
    main()
