import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import color, losses, model, power
from src.pipeline import load_image, sample_images_per_scene, validate_config


class ContractTest(unittest.TestCase):
    def test_factor_features_and_near_identity_initialization(self):
        image = torch.full((16, 16, 3), 0.5)
        features = model.factor_features(image)
        self.assertEqual(features.shape, (16, 16, 3))
        self.assertTrue(torch.allclose(features[..., 1], torch.zeros(16, 16)))
        output = model.FactorModel(target_alpha=1.0)(image)
        self.assertLess(float((output - image).abs().max()), 0.01)

    def test_texture_detects_an_edge(self):
        image = torch.zeros(16, 16, 3)
        image[:, 8:] = 1
        self.assertGreater(float(model.factor_features(image)[..., 1].max()), 0.1)

    def test_lut_matches_model(self):
        torch.manual_seed(0)
        image = torch.rand(16, 16, 3)
        network = model.FactorModel()
        lut = model.generate_lut(network, 16)
        self.assertTrue(torch.allclose(network(image), model.apply_lut(lut, image), atol=2e-4))

    def test_power_target_and_gradient(self):
        original = torch.full((8, 8, 3), 0.5)
        optimized = color.linear_to_srgb(color.srgb_to_linear(original) * 0.8).requires_grad_()
        value = losses.power_loss(original, optimized, 0.8)
        self.assertLess(float(value), 1e-10)
        value.backward()
        self.assertTrue(torch.isfinite(optimized.grad).all())

    def test_config_and_image_loading(self):
        config = json.loads((Path(__file__).parents[1] / "configs/train_config.json").read_text())
        validate_config(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene").mkdir()
            Image.new("RGB", (20, 10)).save(root / "scene/image.png")
            self.assertEqual(len(sample_images_per_scene(root, 1)), 1)
            self.assertEqual(load_image(root / "scene/image.png", 8).shape, (8, 8, 3))

    def test_reporting_power_uses_channel_weights(self):
        red = torch.zeros(4, 4, 3); red[..., 0] = 0.5
        blue = torch.zeros(4, 4, 3); blue[..., 2] = 0.5
        self.assertAlmostEqual(float(power.dynamic_power(red)), float(power.dynamic_power(blue)), places=6)
        self.assertGreater(float(power.display_power(blue)), float(power.display_power(red)))


if __name__ == "__main__":
    unittest.main()
