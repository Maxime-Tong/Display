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
from src.losses import evaluation_metam
from src.power import power_saving
from src.scenes import IMAGE_EXTENSIONS, load_scene_manifest, match_scene


def load_rgb(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)


def evaluate_image(original, optimized, metam, power_weights):
    with torch.no_grad():
        metam_value = metam(optimized.permute(2, 0, 1).unsqueeze(0), original.permute(2, 0, 1).unsqueeze(0), gaze=[0.5, 0.5])
    a = original.detach().cpu().numpy()
    b = optimized.detach().cpu().numpy()
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
    parser.add_argument("--match-distance-threshold", type=float, default=1.8, help="Use the base model/LUT above this DKL distance; <= 0 disables switching")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--save", action="store_true", help="Save optimized images")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--power-weights", nargs=3, type=float, default=[0.22970384, 0.24373232, 0.5265638])
    parser.add_argument("--device", default=None, help="PyTorch device; defaults to CUDA when available")
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("--max-images must be zero or positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"evaluation device: {device}")

    paths = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    paths = paths[:args.max_images] if args.max_images else paths
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, device) if args.model else None
    lut = torch.load(args.lut, map_location=device)["lut"].to(device) if args.lut else None
    manifest = load_scene_manifest(args.scene_manifest) if args.scene_manifest else None
    metam = evaluation_metam(device)
    rows = []
    for current, path in enumerate(paths, 1):
        original = load_rgb(path).to(device)
        cluster_id = matched_cluster_id = None
        match_distance = None
        used_fallback = False
        if manifest:
            matched_cluster_id, match_distance = match_scene(original.detach().cpu().numpy(), manifest, return_distance=True)
            used_fallback = args.match_distance_threshold > 0 and match_distance > args.match_distance_threshold
            cluster_id = -1 if used_fallback else matched_cluster_id
            if used_fallback and model:
                with torch.no_grad():
                    optimized = model(original)
            elif used_fallback:
                optimized = apply_lut(lut, original)
            else:
                lut_path = Path(args.scene_manifest).parent / manifest["lut_paths"][cluster_id]
                active_lut = torch.load(lut_path, map_location=device)["lut"].to(device)
                optimized = apply_lut(active_lut, original)
        elif model:
            with torch.no_grad():
                optimized = model(original)
        else:
            optimized = apply_lut(lut, original)
        optimized_np = optimized.detach().cpu().numpy()
        if args.save:
            relative = path.relative_to(Path(args.data_dir))
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((optimized_np * 255).round().clip(0, 255).astype(np.uint8)).save(destination)
        row = {"filename": path.name, **evaluate_image(original, optimized, metam, args.power_weights)}
        if cluster_id is not None:
            row.update(cluster_id=cluster_id, matched_cluster_id=matched_cluster_id, match_distance=match_distance, used_fallback=used_fallback)
        rows.append(row)
        prefix = f"[{current}/{len(paths)}] {path.name}"
        if cluster_id is not None:
            prefix += f" cluster_id={cluster_id} matched_cluster_id={matched_cluster_id} distance={match_distance:.6f} fallback={used_fallback}"
        metrics = " ".join(f"{key}={row[key]:.6f}" for key in ("power_saving", "psnr", "ssim", "metam"))
        print(f"{prefix} {metrics}")
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
