"""Offline MetaM optimization with a fixed power budget at every LUT node."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .color import linear_to_srgb, srgb_to_linear
from .model import apply_lut
from .power import DISPLAY_POWER_WEIGHTS, power_saving
from .scenes import IMAGE_EXTENSIONS, load_image
from .losses import evaluation_metam


def budget_lut(logits, saving=0.17):
    """Allocate channel reductions offline; preserve weighted power per node.

    Three capped allocation passes suffice for three channels. Zero logits give
    uniform linear dimming. The exported tensor needs only trilinear lookup.
    """
    if logits.ndim != 4 or logits.shape[-1] != 3 or len(set(logits.shape[:3])) != 1 or logits.shape[0] < 2:
        raise ValueError("logits must have shape (N, N, N, 3), N >= 2")
    if not 0 <= saving < 1:
        raise ValueError("saving must be in [0, 1)")
    axis = torch.linspace(0, 1, logits.shape[0], device=logits.device, dtype=logits.dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    weights = logits.new_tensor(DISPLAY_POWER_WEIGHTS)
    energy = srgb_to_linear(grid) * weights
    remaining = saving * energy.sum(-1, keepdim=True)
    capacity = energy
    preference = logits.clamp(-8, 8).exp()
    for _ in range(3):
        scores = energy * preference * (capacity > 0)
        reduction = torch.minimum(capacity, remaining * scores / scores.sum(-1, keepdim=True).clamp_min(1e-20))
        capacity = (capacity - reduction).clamp_min(0)
        remaining = (remaining - reduction.sum(-1, keepdim=True)).clamp_min(0)
    linear = capacity / weights
    # Avoid the infinite derivative of pow at zero in the unused sRGB branch.
    return torch.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * linear.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def split_paths(root, seed, counts):
    # ponytail: image-disjoint sampling within folders; use video/scene groups
    # and official partitions for claims about unseen-scene generalization.
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for scene in sorted(Path(root).iterdir()):
        if not scene.is_dir():
            continue
        paths = sorted(p for p in scene.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            continue
        if len(paths) < sum(counts):
            raise ValueError(f"not enough images in {scene}")
        chosen = rng.sample(paths, sum(counts))
        offset = 0
        for name, count in zip(splits, counts):
            splits[name].extend(chosen[offset:offset + count])
            offset += count
    if not splits["train"]:
        raise ValueError("no scene images found")
    return splits


def measure(lut, images, metric):
    if not images:
        raise ValueError("evaluation split is empty")
    rows = []
    with torch.no_grad():
        for path, image in images:
            output = apply_lut(lut, image)
            value = metric(output.permute(2, 0, 1)[None], image.permute(2, 0, 1)[None], gaze=[0.5, 0.5])
            rows.append({"path": str(path), "power_saving": float(power_saving(image, output)), "metam": float(value)})
    return {"summary": {"metam": float(np.mean([r["metam"] for r in rows])),
                        "power_saving": float(np.mean([r["power_saving"] for r in rows])),
                        "min_power_saving": min(r["power_saving"] for r in rows),
                        "max_power_saving": max(r["power_saving"] for r in rows)}, "images": rows}


def materialize(paths, image_size, device, limit=0):
    """Load only a bounded validation/test window; training stays streaming."""
    selected = paths[:limit] if limit else paths
    return [(path, torch.from_numpy(load_image(path, image_size)).to(device)) for path in selected]


def calibrate_lut(lut, images, saving):
    """Offline scalar calibration on validation images, never test images."""
    axis = torch.linspace(0, 1, lut.shape[0], device=lut.device)
    identity = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    if not torch.isfinite(lut).all() or (lut < 0).any() or (lut > identity + 1e-6).any():
        raise ValueError("baseline violates the multiplicative output-color LUT contract")
    def candidate(strength):
        if strength <= 1:
            return torch.lerp(identity, lut, strength)
        return linear_to_srgb(srgb_to_linear(lut) * (2 - strength))
    low, high = 0., 2.
    with torch.no_grad():
        for _ in range(24):
            middle = (low + high) / 2
            current = candidate(middle)
            achieved = np.mean([float(power_saving(image, apply_lut(current, image))) for _, image in images])
            if achieved < saving:
                low = middle
            else:
                high = middle
    return candidate(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=None, help="JSON config; CLI values override it")
    parser.add_argument("--output-dir", default="outputs/optimized_lut")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--samples", type=int, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--saving", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--baseline-lut", default="base_lut.pt")
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse saved split/LUT; image-size may change for resolution checks")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    defaults = {"steps": 400, "resolution": 33, "image_size": 256, "samples": [8, 2, 3], "val_limit": 200, "test_limit": 0, "seed": 17, "lr": 0.03, "saving": 0.17, "device": "cuda"}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, config.get(name, default))
    if args.steps < 1 or args.resolution < 2 or args.image_size < 64 or args.samples[0] < 1 or args.samples[1] < 1 or min(args.samples) < 0 or args.lr <= 0 or not 0 <= args.saving < 1:
        parser.error("invalid training dimensions, samples, learning rate, or saving")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        splits = json.loads((output / "split.json").read_text())
        metric = evaluation_metam(torch.device(args.device))
        images = {"validation": materialize(splits["validation"], args.image_size, args.device, args.val_limit),
                  "test": materialize(splits["test"], args.image_size, args.device, args.test_limit)}
        checkpoint = torch.load(output / "base_lut.pt", map_location=args.device)
        optimized = checkpoint["lut"]
        saving = checkpoint["saving"]
        candidates = {"uniform": budget_lut(torch.zeros_like(optimized), saving), "optimized": optimized}
        if Path(args.baseline_lut).is_file():
            existing = torch.load(args.baseline_lut, map_location=args.device)["lut"]
            try:
                calibrated = calibrate_lut(existing, images["validation"], saving)
            except ValueError as error:
                baseline_note = str(error)
                print(baseline_note, flush=True)
            else:
                candidates.update(existing=existing, existing_calibrated=calibrated)
                baseline_note = "Baseline calibrated on validation images only."
        else:
            baseline_note = "No baseline LUT supplied."
        report = {"image_size": args.image_size, "saving": saving, "test": {}}
        report["baseline_note"] = baseline_note
        if images["test"]:
            for name, lut in candidates.items():
                report["test"][name] = measure(lut, images["test"], metric)
                print(name, report["test"][name]["summary"], flush=True)
        else:
            report["test_note"] = "No internal held-out test split; use eval.py on the target dataset."
        (output / f"metrics_{args.image_size}.json").write_text(json.dumps(report, indent=2))
        return
    splits = split_paths(args.data_dir, args.seed, args.samples)
    (output / "split.json").write_text(json.dumps({k: list(map(str, v)) for k, v in splits.items()}, indent=2))
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    images = {"validation": materialize(splits["validation"], args.image_size, args.device, args.val_limit),
              "test": materialize(splits["test"], args.image_size, args.device, args.test_limit)}
    metric = evaluation_metam(torch.device(args.device))
    logits = torch.nn.Parameter(torch.zeros(args.resolution, args.resolution, args.resolution, 3, device=args.device))
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    uniform = budget_lut(logits, args.saving).detach()
    best = measure(uniform, images["validation"], metric)["summary"]["metam"]
    best_lut = uniform.clone()
    best_step = 0
    history = []
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        path = rng.choice(splits["train"])
        image = torch.from_numpy(load_image(path, args.image_size)).to(args.device)
        optimizer.zero_grad()
        lut = budget_lut(logits, args.saving)
        adapted = apply_lut(lut, image)
        loss = metric(adapted.permute(2, 0, 1)[None], image.permute(2, 0, 1)[None], gaze=[0.5, 0.5])
        (loss * 1000).backward()
        if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():
            raise RuntimeError("nonfinite training loss/gradient")
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            candidate = budget_lut(logits, args.saving).detach()
            validation = measure(candidate, images["validation"], metric)["summary"]
            if validation["metam"] < best:
                best, best_lut = validation["metam"], candidate.clone()
                best_step = step
            history.append({"step": step, **validation})
            print(f"step {step}: {validation}", flush=True)
            torch.save({"lut": best_lut.cpu(), "saving": args.saving, "power_weights": DISPLAY_POWER_WEIGHTS,
                        "representation": "output_srgb", "best_step": best_step}, output / "base_lut.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2))
    report = {"config": vars(args), "selection": "lowest validation MetaM; test unused until final evaluation", "best_step": best_step, "test": {}}
    candidates = {"uniform": uniform, "optimized": best_lut}
    if Path(args.baseline_lut).is_file():
        existing = torch.load(args.baseline_lut, map_location=args.device)["lut"]
        try:
            calibrated = calibrate_lut(existing, images["validation"], args.saving)
        except ValueError as error:
            report["baseline_note"] = str(error)
        else:
            candidates.update(existing=existing, existing_calibrated=calibrated)
    if images["test"]:
        for name, lut in candidates.items():
            report["test"][name] = measure(lut, images["test"], metric)
            print(name, report["test"][name]["summary"], flush=True)
    else:
        report["test_note"] = "No internal held-out test split; use eval.py on the target dataset."
    (output / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
