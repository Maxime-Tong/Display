from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from _common import image_paths, load_rgb
from src.color import linear_to_srgb, srgb_to_linear
from src.perception import MetamericLoss
from src.power import power_saving


def main():
    parser = argparse.ArgumentParser(description="Evaluate uniform dimming baseline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/uniform")
    parser.add_argument("--target-alpha", type=float, default=0.83)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--power-weights", nargs=3, type=float, default=[0.22970384, 0.24373232, 0.5265638])
    parser.add_argument("--device")
    args = parser.parse_args()
    if not 0 < args.target_alpha <= 1:
        parser.error("target-alpha must be in (0, 1]")
    if args.max_images < 0:
        parser.error("max-images must be non-negative")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = image_paths(args.data_dir, args.max_images)
    if not paths:
        raise ValueError("dataset contains no images")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metam = MetamericLoss(device=device, real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=5.0, mode="quadratic", loss_type="L1", use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4, use_radial_weight=True)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for current, path in enumerate(paths, 1):
        original = load_rgb(path, device)
        # target_alpha is a linear-RGB power fraction, not an sRGB multiplier.
        optimized = linear_to_srgb(args.target_alpha * srgb_to_linear(original))
        with torch.no_grad():
            metam_value = metam(optimized.permute(2, 0, 1)[None], original.permute(2, 0, 1)[None], gaze=[0.5, 0.5])
            lpips_value = lpips_model(optimized.permute(2, 0, 1)[None] * 2 - 1, original.permute(2, 0, 1)[None] * 2 - 1).item()
        a, b = original.cpu().numpy(), optimized.cpu().numpy()
        row = {
            "filename": str(path.relative_to(args.data_dir)),
            "power_saving": float(power_saving(original, optimized, args.power_weights)),
            "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
            "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
            "metam": float(metam_value),
            "lpips": lpips_value,
        }
        rows.append(row)
        print(f"[{current}/{len(paths)}] {path.name} " + " ".join(f"{k}={row[k]:.6f}" for k in ("power_saving", "psnr", "ssim", "metam", "lpips")))

    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("power_saving", "psnr", "ssim", "metam", "lpips")}
    (output / "metrics.json").write_text(json.dumps({"config": {"target_alpha": args.target_alpha}, "images": rows, "summary": summary}, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("summary", summary)


if __name__ == "__main__":
    main()
