from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.model import apply_lut, load_model
from src.perception import MetamericLoss
from src.pipeline import IMAGE_EXTENSIONS
from src.power import power_saving


def load_rgb(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scalar factor model or LUT")
    parser.add_argument("--data-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--lut")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model, device) if args.model else None
    asset = torch.load(args.lut, map_location=device) if args.lut else None
    lut = asset["lut"].to(device) if asset else None
    paths = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError("dataset contains no images")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metam = MetamericLoss(device=device, real_image_width=1.4, real_viewing_distance=0.7,
                          equi=False, alpha=5.0, mode="quadratic", loss_type="L1",
                          use_l2_foveal_loss=False, n_pyramid_levels=5,
                          n_orientations=4, use_radial_weight=True)
    rows = []
    for path in paths:
        original = load_rgb(path).to(device)
        with torch.no_grad():
            optimized = model(original) if model else apply_lut(lut, original)
            metam_value = metam(optimized.permute(2, 0, 1).unsqueeze(0),
                                original.permute(2, 0, 1).unsqueeze(0), gaze=[0.5, 0.5])
        a, b = original.cpu().numpy(), optimized.cpu().numpy()
        row = {"filename": path.name,
               "power_saving": float(power_saving(original, optimized)),
               "psnr": float(peak_signal_noise_ratio(a, b, data_range=1)),
               "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1)),
               "metam": float(metam_value)}
        rows.append(row)
        if args.save:
            Image.fromarray((b * 255).round().clip(0, 255).astype(np.uint8)).save(output / path.name)
        print(path.name, " ".join(f"{key}={row[key]:.6f}" for key in row if key != "filename"))
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in ("power_saving", "psnr", "ssim", "metam")}
    (output / "metrics.json").write_text(json.dumps({"images": rows, "summary": summary}, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print("summary", summary)


if __name__ == "__main__":
    main()
