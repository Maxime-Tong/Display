import unittest

import torch

from src.color import linear_to_srgb, srgb_to_linear
from src.model import apply_lut
from src.optimize_lut import budget_lut, calibrate_lut
from src.power import DISPLAY_POWER_WEIGHTS, power_saving


class BudgetLutTest(unittest.TestCase):
    def test_offline_calibration_and_invalid_legacy_asset(self):
        torch.manual_seed(7)
        images = [("sample", torch.rand(8, 8, 3))]
        lut = budget_lut(torch.zeros(9, 9, 9, 3), saving=0.25)
        calibrated = calibrate_lut(lut, images, saving=0.17)
        self.assertAlmostEqual(float(power_saving(images[0][1], apply_lut(calibrated, images[0][1]))), 0.17, places=6)
        with self.assertRaisesRegex(ValueError, "contract"):
            calibrate_lut(torch.full_like(lut, 0.9), images, saving=0.17)

    def test_native_sampler_matches_eight_corners(self):
        torch.manual_seed(3)
        lut = torch.rand(7, 7, 7, 3, requires_grad=True)
        image = torch.rand(13, 11, 3) * 1.2 - 0.1
        position = image.clamp(0, 1) * 6
        low = position.floor().long()
        fraction = position - low
        expected = torch.zeros_like(image)
        for r in (0, 1):
            for g in (0, 1):
                for b in (0, 1):
                    corner = torch.tensor([r, g, b])
                    index = (low + corner).clamp(max=6)
                    weight = torch.where(corner.bool(), fraction, 1 - fraction).prod(-1)
                    expected = expected + lut[index[..., 0], index[..., 1], index[..., 2]] * weight[..., None]
        actual = apply_lut(lut, image)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)
        expected_grad = torch.autograd.grad(expected.sum(), lut, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual.sum(), lut)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)

    def test_budget_bounds_gradients_and_lookup(self):
        torch.manual_seed(17)
        logits = (torch.randn(9, 9, 9, 3) * 4).requires_grad_()
        axis = torch.linspace(0, 1, 9)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
        weights = torch.tensor(DISPLAY_POWER_WEIGHTS)
        lut = budget_lut(logits)
        torch.testing.assert_close((srgb_to_linear(lut) * weights).sum(-1),
                                   0.83 * (srgb_to_linear(grid) * weights).sum(-1), atol=2e-7, rtol=2e-6)
        self.assertTrue((lut >= 0).all() and (lut <= grid + 1e-6).all())
        lut.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        uniform = budget_lut(torch.zeros_like(logits))
        torch.testing.assert_close(uniform, linear_to_srgb(srgb_to_linear(grid) * 0.83))
        torch.testing.assert_close(apply_lut(lut, grid.reshape(81, 9, 3)), lut.reshape(81, 9, 3))
        self.assertTrue((apply_lut(lut, torch.zeros(2, 2, 3)) == 0).all())
        with self.assertRaises(ValueError):
            budget_lut(logits, 1.1)


if __name__ == "__main__":
    unittest.main()
