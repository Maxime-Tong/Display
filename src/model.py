from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .color import srgb_to_linear


def model_features(image, tile_size=4):
    """Return normalized luminance, tile variance, and position features."""
    if image.shape[-1] != 3:
        raise ValueError("image must have three channels")
    linear = srgb_to_linear(image.clamp(0, 1))
    luminance = linear @ image.new_tensor([0.2126, 0.7152, 0.0722])
    h, w = image.shape[-3:-1]
    pad_h, pad_w = (-h) % tile_size, (-w) % tile_size
    tiles = F.pad(luminance[None, None], (0, pad_w, 0, pad_h), mode="replicate")
    mean = F.avg_pool2d(tiles, tile_size, tile_size)
    variance = F.avg_pool2d(tiles.square(), tile_size, tile_size) - mean.square()
    variance = variance.repeat_interleave(tile_size, -2).repeat_interleave(tile_size, -1)[..., :h, :w][0, 0]
    y = torch.linspace(0, 1, h, device=image.device, dtype=image.dtype)[:, None]
    x = torch.linspace(0, 1, w, device=image.device, dtype=image.dtype)[None, :]
    position = (x + y) / 2
    return torch.stack((luminance, (variance * 4).clamp(0, 1), position.expand(h, w)), -1)


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
