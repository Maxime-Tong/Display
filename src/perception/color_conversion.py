"""Minimal color conversion helpers extracted from ``odak.learn.perception.color_conversion``.

Only the functions needed by the metameric losses are kept.  The odak logger
dependency is intentionally omitted.
"""

import torch


def rgb_2_ycrcb(image):
    """
    Converts an image from RGB colourspace to YCrCb colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be an RGB floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------

    ycrcb   : torch.tensor
              Image converted to YCrCb colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    ycrcb = torch.zeros(image.size()).to(image.device)
    ycrcb[:, 0, :, :] = (
        0.299 * image[:, 0, :, :]
        + 0.587 * image[:, 1, :, :]
        + 0.114 * image[:, 2, :, :]
    )
    ycrcb[:, 1, :, :] = 0.5 + 0.713 * (image[:, 0, :, :] - ycrcb[:, 0, :, :])
    ycrcb[:, 2, :, :] = 0.5 + 0.564 * (image[:, 2, :, :] - ycrcb[:, 0, :, :])
    return ycrcb


def ycrcb_2_rgb(image):
    """
    Converts an image from YCrCb colourspace to RGB colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be a YCrCb floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------
    rgb     : torch.tensor
              Image converted to RGB colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    rgb = torch.zeros(image.size(), device=image.device)
    rgb[:, 0, :, :] = image[:, 0, :, :] + 1.403 * (image[:, 1, :, :] - 0.5)
    rgb[:, 1, :, :] = (
        image[:, 0, :, :]
        - 0.714 * (image[:, 1, :, :] - 0.5)
        - 0.344 * (image[:, 2, :, :] - 0.5)
    )
    rgb[:, 2, :, :] = image[:, 0, :, :] + 1.773 * (image[:, 2, :, :] - 0.5)
    return rgb
