from __future__ import annotations

from .color import srgb_to_linear

DISPLAY_POWER_WEIGHTS = (0.22970384, 0.24373232, 0.5265638)


def dynamic_power(image):
    """ML-PEA effective dynamic power: sum of linear-RGB channel means."""
    linear = srgb_to_linear(image)
    return linear[..., 0].mean() + linear[..., 1].mean() + linear[..., 2].mean()


def target_power(image, alpha):
    return alpha * dynamic_power(image)


def display_power(image, weights=DISPLAY_POWER_WEIGHTS):
    """Weighted linear-RGB power used only for reporting."""
    linear = srgb_to_linear(image)
    return sum(linear[..., channel].mean() * weight for channel, weight in enumerate(weights))


def power_saving(original, optimized, weights=DISPLAY_POWER_WEIGHTS):
    """Reported weighted display-power saving; never used by the loss."""
    original_power = display_power(original, weights)
    if float(original_power) <= 0:
        return 0.0
    return 1.0 - display_power(optimized, weights) / original_power
