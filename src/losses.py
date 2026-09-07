from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .perception import MetamericLossUniform
from .color import srgb_to_linear


@dataclass(frozen=True)
class LossConfig:
    target_alpha: float = 0.8
    lambda_power: float = 50.0
    lambda_metam: float = 0.5
    lambda_weber: float = 0.0
    lambda_ssim: float = 5.0
    weber_epsilon: float = 0.01
    power_weights: tuple[float, float, float] = (0.22970384, 0.24373232, 0.5265638)


def power_loss(original, optimized, target_alpha=0.8):
    """Match retained linear-RGB power independently in each 4x4 tile."""
    def tile_power(image):
        linear = srgb_to_linear(image)
        h, w = image.shape[:2]
        pad_h, pad_w = (-h) % 4, (-w) % 4
        padded = F.pad(linear.permute(2, 0, 1)[None], (0, pad_w, 0, pad_h), mode="replicate")
        return F.avg_pool2d(padded, 4, 4).sum(dim=1)

    return (tile_power(optimized) - target_alpha * tile_power(original)).square().mean()


def weber_loss(original, optimized, weights=(0.22970384, 0.24373232, 0.5265638), epsilon=0.01):
    weights = torch.as_tensor(weights, dtype=original.dtype, device=original.device)
    reference = (original * weights).sum(dim=-1)
    value = (optimized * weights).sum(dim=-1)
    return ((value - reference).abs() / (reference + epsilon)).mean()


def _ssim_value(x, y, window_size=11, sigma=1.5):
    if x.ndim == 3:
        x, y = x.permute(2, 0, 1).unsqueeze(0), y.permute(2, 0, 1).unsqueeze(0)
    axis = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    kernel = torch.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel = (kernel[:, None] * kernel[None, :]) / kernel.sum() ** 2
    kernel = kernel.expand(x.shape[1], 1, -1, -1)
    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=pad, groups=y.shape[1])
    var_x = F.conv2d(x * x, kernel, padding=pad, groups=x.shape[1]) - mu_x ** 2
    var_y = F.conv2d(y * y, kernel, padding=pad, groups=y.shape[1]) - mu_y ** 2
    cov = F.conv2d(x * y, kernel, padding=pad, groups=x.shape[1]) - mu_x * mu_y
    return (((2 * mu_x * mu_y + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) /
            ((mu_x ** 2 + mu_y ** 2 + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2)).clamp_min(1e-10)).mean()


def ssim_loss(original, optimized):
    return 1.0 - _ssim_value(original, optimized)


class CombinedLoss(torch.nn.Module):
    def __init__(self, config=LossConfig(), device="cpu"):
        super().__init__()
        self.config = config
        self.metam = MetamericLossUniform(
            n_pyramid_levels=5, n_orientations=4, pooling_size=64,
            device=device, loss_type="L1",
        )

    def forward(self, original, optimized):
        cfg = self.config
        metam_original = original.permute(2, 0, 1).unsqueeze(0)
        metam_optimized = optimized.permute(2, 0, 1).unsqueeze(0)
        parts = {
            "power": power_loss(original, optimized, cfg.target_alpha),
            "metam": self.metam(metam_optimized, metam_original),
            "weber": weber_loss(original, optimized, cfg.power_weights, cfg.weber_epsilon),
            "ssim": ssim_loss(original, optimized),
        }
        parts["total"] = (cfg.lambda_power * parts["power"] + cfg.lambda_metam * parts["metam"]
                          + cfg.lambda_weber * parts["weber"] + cfg.lambda_ssim * parts["ssim"])
        return parts


def combined_loss(original, optimized, config=LossConfig(), device="cpu"):
    return CombinedLoss(config, device)(original, optimized)
