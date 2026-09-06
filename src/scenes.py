from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .color import rgb_to_dkl, srgb_to_linear


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_image(path, size=128):
    image = Image.open(path).convert("RGB")
    if size:
        image.thumbnail((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_dkl_feature(image):
    """Original fast DKL feature: channel mean, std, and median."""
    dkl = rgb_to_dkl(srgb_to_linear(np.asarray(image))).reshape(-1, 3)
    return np.concatenate([dkl.mean(0), dkl.std(0), np.median(dkl, axis=0)]).astype(np.float32)


def cluster_dkl_scenes(features, number_of_clusters, seed=0, max_iter=100):
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2 or not 0 < number_of_clusters <= len(data):
        raise ValueError("features must be 2D and clusters must not exceed samples")
    mean, std = data.mean(0), data.std(0)
    std[std < 1e-6] = 1
    normalized = (data - mean) / std
    rng = np.random.default_rng(seed)
    centers = normalized[rng.choice(len(data), number_of_clusters, replace=False)].copy()
    for _ in range(max_iter):
        labels = np.linalg.norm(normalized[:, None] - centers[None, :], axis=2).argmin(1)
        updated = np.stack([normalized[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(number_of_clusters)])
        if np.allclose(updated, centers):
            break
        centers = updated
    labels = np.linalg.norm(normalized[:, None] - centers[None, :], axis=2).argmin(1)
    return labels, centers, mean.astype(np.float32), std.astype(np.float32)


def save_scene_manifest(path, centers, mean, std, lut_paths):
    payload = {"feature": "dkl_mean_std_median", "centers": np.asarray(centers).tolist(), "mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist(), "lut_paths": [str(p) for p in lut_paths]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scene_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def match_scene(image, manifest):
    feature = (extract_dkl_feature(image) - np.asarray(manifest["mean"])) / np.asarray(manifest["std"])
    return int(np.linalg.norm(np.asarray(manifest["centers"]) - feature, axis=1).argmin())
