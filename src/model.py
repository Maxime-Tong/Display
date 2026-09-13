from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

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
    # grid_sample coordinates are (W, H, D), hence BGR for an RGB-indexed LUT.
    coordinates = (image.clamp(0, 1).flip(-1) * 2 - 1)[None, None]
    volume = lut.permute(3, 0, 1, 2)[None]
    sampled = F.grid_sample(volume, coordinates, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return sampled[0, :, 0].permute(1, 2, 0).clamp(0, 1)
