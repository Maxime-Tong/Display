"""Self-contained metameric perceptual losses.

Extracted from ``odak.learn.perception`` so that ``screen_adaptor`` no longer
needs the ``odak`` package installed.  Only the classes/functions actually used
by ``screen_adaptor`` are kept, and the odak logging component is excluded.
"""

from .metameric_loss import MetamericLoss
from .metameric_loss_uniform import MetamericLossUniform

__all__ = ["MetamericLoss", "MetamericLossUniform"]
