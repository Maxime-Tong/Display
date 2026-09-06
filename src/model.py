from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .color import rgb_to_dkl, srgb_to_linear


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
        gain = (torch.tanh(self.network(rgb_to_dkl(srgb_to_linear(image)))) + 1) / 2
        return (image * gain).clamp(0, 1)


def apply_model(model, image):
    return model(image)


def generate_lut(model, resolution=16, device="cpu"):
    axis = torch.linspace(0, 1, resolution, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    model = model.to(device).eval()
    with torch.no_grad():
        return model(grid).cpu()


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": model.hidden_dim, "depth": model.depth}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = ColorModel(checkpoint["hidden_dim"], checkpoint["depth"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def apply_lut(lut, image):
    """Trilinear LUT interpolation for HWC sRGB tensors."""
    size = lut.shape[0]
    position = image.clamp(0, 1) * (size - 1)
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
    return out.clamp(0, 1)
