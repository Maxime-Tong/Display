from __future__ import annotations

import numpy as np
import torch


RGB_TO_DKL = np.array([
    [0.14376143, 0.16556473, 0.00228754],
    [-0.21244304, -0.71424228, -0.06559154],
    [0.21259150, 0.71517140, 0.07219711],
], dtype=np.float32)

RGB_TO_OKLAB = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
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


def rgb_to_oklab(image):
    """Convert linear RGB (..., 3) to the perceptual OKLab space."""
    if isinstance(image, torch.Tensor):
        rgb_matrix = torch.as_tensor(RGB_TO_OKLAB, dtype=image.dtype, device=image.device)
        lab_matrix = torch.as_tensor(LMS_TO_OKLAB, dtype=image.dtype, device=image.device)
        lms = image @ rgb_matrix.T
        return torch.sign(lms) * torch.abs(lms).pow(1 / 3) @ lab_matrix.T
    lms = np.asarray(image) @ RGB_TO_OKLAB.T
    return np.cbrt(lms) @ LMS_TO_OKLAB.T
