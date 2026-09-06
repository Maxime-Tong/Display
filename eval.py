from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).parent))

from src.model import apply_lut, load_model
from src.perception import MetamericLoss
from src.power import power_saving
from src.scenes import IMAGE_EXTENSIONS, load_scene_manifest, match_scene


def load_rgb(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)


def evaluate_image(original, optimized, metam, power_weights):
    a, b = original.numpy(), optimized.numpy()
    with torch.no_grad():
        metam_value = metam(optimized.permute(2, 0, 1).unsqueeze(0), original.permute(2, 0, 1).unsqueeze(0), gaze=[0.5, 0.5])
    return {
        "power_saving": float(power_saving(original, optimized, power_weights)),
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
        "metam": float(metam_value),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate screen adaptor")
    parser.add_argument("--data-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--lut")
    parser.add_argument("--scene-manifest")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--power-weights", nargs=3, type=float, default=[0.22970384, 0.24373232, 0.5265638])
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    paths = paths[:args.max_images] if args.max_images else paths
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model) if args.model else None
    lut = torch.load(args.lut, map_location="cpu")["lut"] if args.lut else None
    manifest = load_scene_manifest(args.scene_manifest) if args.scene_manifest else None
    metam = MetamericLoss(real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=5.0, mode="quadratic", loss_type="L1", use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4, use_radial_weight=True)
    rows = []
    for path in paths:
        original = load_rgb(path)
        if manifest:
            index = match_scene(original.numpy(), manifest)
            lut_path = Path(args.scene_manifest).parent / manifest["lut_paths"][index]
            active_lut = torch.load(lut_path, map_location="cpu")["lut"]
            optimized = apply_lut(active_lut, original)
        elif model:
            with torch.no_grad():
                optimized = model(original)
        else:
            optimized = apply_lut(lut, original)
        Image.fromarray((optimized.numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(output / path.name)
        row = {"filename": path.name, **evaluate_image(original, optimized, metam, args.power_weights)}
        rows.append(row)
        print(" ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if not rows:
        raise ValueError("dataset contains no images")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("power_saving", "psnr", "ssim", "metam")}
    (output / "metrics.json").write_text(json.dumps({"images": rows, "summary": summary}, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("summary", summary)


if __name__ == "__main__":
    main()
