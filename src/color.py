from __future__ import annotations

import numpy as np
import torch


RGB_TO_DKL = np.array([
    [0.14376143, 0.16556473, 0.00228754],
    [-0.21244304, -0.71424228, -0.06559154],
    [0.21259150, 0.71517140, 0.07219711],
], dtype=np.float32)


def srgb_to_linear(image):
    """Convert an sRGB NumPy array or tensor in [0, 1] to linear RGB."""
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, 1)
        return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    image = np.clip(np.asarray(image), 0, 1)
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image):
    """Convert a linear-RGB NumPy array or tensor in [0, 1] to sRGB."""
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, 1)
        return torch.where(image <= 0.0031308, 12.92 * image, 1.055 * image ** (1 / 2.4) - 0.055)
    image = np.clip(np.asarray(image), 0, 1)
    return np.where(image <= 0.0031308, 12.92 * image, 1.055 * image ** (1 / 2.4) - 0.055)


def rgb_to_dkl(image):
    """Convert linear RGB (..., 3) to DKL (..., 3)."""
    if isinstance(image, torch.Tensor):
        matrix = torch.as_tensor(RGB_TO_DKL, dtype=image.dtype, device=image.device)
        return image @ matrix.T
    return np.asarray(image) @ RGB_TO_DKL.T
