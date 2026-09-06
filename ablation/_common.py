from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import apply_lut, load_model
from src.perception import MetamericLoss
from src.power import power_saving
from src.scenes import IMAGE_EXTENSIONS, load_scene_manifest, match_scene


def image_paths(data_dir, max_images=0):
    paths = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:max_images] if max_images else paths


def load_rgb(path, device):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def blockwise_lut(lut, image, block_size):
    if block_size == 1:
        return apply_lut(lut, image)
    h, w = image.shape[:2]
    pad_h, pad_w = (-h) % block_size, (-w) % block_size
    mode = "reflect" if h > 1 and w > 1 and pad_h < h and pad_w < w else "replicate"
    padded = F.pad(image.permute(2, 0, 1)[None], (0, pad_w, 0, pad_h), mode=mode)
    average = F.avg_pool2d(padded, block_size, block_size)[0].permute(1, 2, 0)
    transformed = apply_lut(lut, average)
    average = average.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:h, :w]
    transformed = transformed.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:h, :w]
    return (transformed * image / average.clamp_min(1e-6)).clamp(0, 1)


class Assets:
    def __init__(self, model_dir, mode, representation, device, lut_resolution=None, threshold=1.8):
        self.root, self.mode, self.representation = Path(model_dir), mode, representation
        self.device, self.threshold = device, threshold
        self.manifest = load_scene_manifest(self.root / "scene_manifest.json") if mode == "cluster" else None
        prefix = "base" if mode == "single" else "cluster"
        count = 1 if mode == "single" else len(self.manifest["centers"])
        if representation == "net":
            self.items = [load_model(self.root / ("base_checkpoint.pt" if mode == "single" else f"cluster_{i}_checkpoint.pt"), device) for i in range(count)]
        else:
            self.items = []
            for i in range(count):
                path = self.root / ("base_lut.pt" if mode == "single" else self.manifest["lut_paths"][i])
                if lut_resolution:
                    model_path = self.root / ("base_checkpoint.pt" if mode == "single" else f"cluster_{i}_checkpoint.pt")
                    from src.model import generate_lut
                    item = generate_lut(load_model(model_path, device), lut_resolution, device).to(device)
                else:
                    item = torch.load(path, map_location=device)["lut"].to(device)
                self.items.append(item)

    def transform(self, image, block_size):
        index, distance = (0, 0.0) if self.mode == "single" else match_scene(image.detach().cpu().numpy(), self.manifest, True)
        fallback = self.mode == "cluster" and self.threshold > 0 and distance > self.threshold
        if fallback:
            base = self.root / ("base_checkpoint.pt" if self.representation == "net" else "base_lut.pt")
            item = load_model(base, self.device) if self.representation == "net" else torch.load(base, map_location=self.device)["lut"].to(self.device)
        else:
            item = self.items[index]
        with torch.no_grad():
            output = item(image) if self.representation == "net" else blockwise_lut(item, image, block_size)
        return output, (-1 if fallback else index), distance, fallback


def evaluate(model_dir, data_dir, mode, representation, block_size, max_images, device, lut_resolution=None, threshold=1.8):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    assets = Assets(model_dir, mode, representation, device, lut_resolution, threshold)
    metam = MetamericLoss(device=device, real_image_width=1.4, real_viewing_distance=0.7, equi=False, alpha=5.0, mode="quadratic", loss_type="L1", use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4, use_radial_weight=True)
    rows = []
    paths = image_paths(data_dir, max_images)
    if not paths:
        raise ValueError("dataset contains no images")
    for current, path in enumerate(paths, 1):
        original = load_rgb(path, device)
        optimized, cluster_id, distance, fallback = assets.transform(original, block_size)
        a, b = original.cpu().numpy(), optimized.cpu().numpy()
        with torch.no_grad():
            metam_value = metam(optimized.permute(2, 0, 1)[None], original.permute(2, 0, 1)[None], gaze=[0.5, 0.5])
        row = {"filename": str(path.relative_to(data_dir)), "saving": float(power_saving(original, optimized)), "psnr": float(peak_signal_noise_ratio(a, b, data_range=1)), "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1)), "metam": float(metam_value), "cluster_id": cluster_id, "match_distance": distance, "fallback": fallback}
        rows.append(row)
        print(f"[{current}/{len(paths)}] {path.name} cluster_id={cluster_id} saving={row['saving']:.6f} PSNR={row['psnr']:.4f} SSIM={row['ssim']:.6f} MetaM={row['metam']:.6f}")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("saving", "psnr", "ssim", "metam")}
    return rows, summary


def write_result(path, config, rows, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "images": rows, "summary": summary}, indent=2), encoding="utf-8")
