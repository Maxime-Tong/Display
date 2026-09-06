import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import blockwise_lut


class AblationTest(unittest.TestCase):
    def test_identity_lut_preserves_image(self):
        axis = torch.linspace(0, 1, 8)
        lut = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
        image = torch.rand(9, 11, 3)
        for block_size in (1, 4):
            self.assertTrue(torch.allclose(blockwise_lut(lut, image, block_size), image, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
