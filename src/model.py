from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .color import linear_to_srgb, srgb_to_linear


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def factor_features(image):
    """Return HWC features: linear luminance, texture strength, radial position."""
    linear = srgb_to_linear(image)
    luminance = (linear * image.new_tensor(LUMA_WEIGHTS)).sum(-1)
    dx = F.pad((luminance[:, 1:] - luminance[:, :-1]).abs(), (0, 1))
    dy = F.pad((luminance[1:, :] - luminance[:-1, :]).abs(), (0, 0, 0, 1))
    texture = F.avg_pool2d((dx + dy)[None, None], 3, 1, 1)[0, 0].clamp(0, 1)
    y = torch.linspace(-1, 1, image.shape[0], device=image.device, dtype=image.dtype)
    x = torch.linspace(-1, 1, image.shape[1], device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    position = torch.sqrt(xx.square() + yy.square()).div(2 ** 0.5).clamp(0, 1)
    return torch.stack((luminance, texture, position), -1)


class FactorModel(nn.Module):
    def __init__(self, hidden_dim=32, depth=2, max_attenuation=0.5,
                 channel_compensation=(1.0, 1.0, 1.0)):
        super().__init__()
        layers, width = [], 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden_dim), nn.SiLU()]
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.0)
        self.hidden_dim, self.depth = hidden_dim, depth
        self.max_attenuation = float(max_attenuation)
        self.register_buffer("channel_compensation", torch.tensor(channel_compensation))

    def factor(self, features):
        return 1.0 - self.max_attenuation * torch.sigmoid(self.network(features))

    def forward(self, image):
        factor = self.factor(factor_features(image))
        linear = srgb_to_linear(image) * factor * self.channel_compensation
        return linear_to_srgb(linear.clamp(0, 1))


def generate_lut(model, resolution=16, device="cpu"):
    axis = torch.linspace(0, 1, resolution, device=device)
    features = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    model = model.to(device).eval()
    with torch.no_grad():
        return model.factor(features).cpu()


def _interpolate_lut(lut, features):
    size = lut.shape[0]
    position = features.clamp(0, 1) * (size - 1)
    low = position.floor().long()
    high = (low + 1).clamp_max(size - 1)
    fraction = position - low
    out = torch.zeros_like(features[..., :1])
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                index = torch.stack((high[..., 0] if a else low[..., 0],
                                     high[..., 1] if b else low[..., 1],
                                     high[..., 2] if c else low[..., 2]), -1)
                weight = ((fraction[..., 0] if a else 1 - fraction[..., 0]) *
                          (fraction[..., 1] if b else 1 - fraction[..., 1]) *
                          (fraction[..., 2] if c else 1 - fraction[..., 2]))
                out += lut[index[..., 0], index[..., 1], index[..., 2]] * weight[..., None]
    return out


def apply_lut(lut, image, channel_compensation=(1.0, 1.0, 1.0)):
    factor = _interpolate_lut(lut, factor_features(image))
    compensation = image.new_tensor(channel_compensation)
    return linear_to_srgb((srgb_to_linear(image) * factor * compensation).clamp(0, 1))


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": model.hidden_dim,
                "depth": model.depth, "max_attenuation": model.max_attenuation}, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    compensation = checkpoint["state_dict"]["channel_compensation"].tolist()
    model = FactorModel(checkpoint["hidden_dim"], checkpoint["depth"],
                        checkpoint["max_attenuation"], compensation).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
