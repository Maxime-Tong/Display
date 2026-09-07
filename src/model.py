from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .color import rgb_to_oklab, srgb_to_linear


def model_features(image, tile_size=4):
    """Return OKLab brightness, saturation, and 4x4 texture density."""
    if image.shape[-1] != 3:
        raise ValueError("image must have three channels")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    linear = srgb_to_linear(image.clamp(0, 1))
    oklab = rgb_to_oklab(linear)
    brightness = oklab[..., 0].clamp(0, 1)
    chroma = torch.linalg.vector_norm(oklab[..., 1:], dim=-1)
    chroma = torch.where(chroma < 1e-6, torch.zeros_like(chroma), chroma)
    saturation = 1 - torch.exp(-chroma / (brightness + 0.05) / 0.5)
    h, w = image.shape[-3:-1]
    pad_h, pad_w = (-h) % tile_size, (-w) % tile_size
    dx = F.pad((brightness[:, 1:] - brightness[:, :-1]).abs(), (0, 1))
    dy = F.pad((brightness[1:] - brightness[:-1]).abs(), (0, 0, 0, 1))
    texture = F.pad((dx.square() + dy.square()).sqrt()[None, None], (0, pad_w, 0, pad_h), mode="replicate")
    texture = F.avg_pool2d(texture, tile_size, tile_size)
    texture = texture.repeat_interleave(tile_size, -2).repeat_interleave(tile_size, -1)[..., :h, :w][0, 0]
    texture = 1 - torch.exp(-texture / 0.05)
    return torch.stack((brightness, saturation.clamp(0, 1), texture.clamp(0, 1)), -1)


class ColorModel(nn.Module):
    def __init__(self, hidden_dim=32, depth=2):
        super().__init__()
        layers = []
        width = 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden_dim), nn.SiLU()]
            width = hidden_dim
        self.network = nn.Sequential(*layers, nn.Linear(width, 3))
        self.hidden_dim, self.depth = hidden_dim, depth

    def forward(self, image):
        gain = self.gain(model_features(image))
        return (image * gain).clamp(0, 1)

    def gain(self, features):
        return (torch.tanh(self.network(features)) + 1) / 2


def apply_model(model, image):
    return model(image)


def generate_lut(model, resolution=16, device="cpu"):
    axis = torch.linspace(0, 1, resolution, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    model = model.to(device).eval()
    with torch.no_grad():
        return model.gain(grid).cpu()


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": model.hidden_dim, "depth": model.depth}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = ColorModel(checkpoint["hidden_dim"], checkpoint["depth"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def apply_lut(lut, image, tile_size=4):
    """Apply a feature LUT to an HWC sRGB image."""
    size = lut.shape[0]
    position = model_features(image, tile_size).clamp(0, 1) * (size - 1)
    low = position.floor().long().clamp(0, size - 1)
    high = (low + 1).clamp(0, size - 1)
    fraction = position - low
    out = torch.zeros_like(image)
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                index = torch.stack([high[..., 0] if r else low[..., 0], high[..., 1] if g else low[..., 1], high[..., 2] if b else low[..., 2]], -1)
                weight = ((fraction[..., 0] if r else 1 - fraction[..., 0]) * (fraction[..., 1] if g else 1 - fraction[..., 1]) * (fraction[..., 2] if b else 1 - fraction[..., 2]))
                out += lut[index[..., 0], index[..., 1], index[..., 2]] * weight[..., None]
    return (image * out).clamp(0, 1)
