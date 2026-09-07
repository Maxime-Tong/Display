import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from src import color, losses, model, power, scenes
from src.pipeline import sample_images_per_scene, validate_config


class ContractTest(unittest.TestCase):
    def test_public_functions_exist(self):
        required = {
            color: ["srgb_to_linear", "linear_to_srgb", "rgb_to_dkl"],
            power: ["dynamic_power", "target_power", "display_power", "power_saving"],
            losses: ["power_loss", "weber_loss", "ssim_loss", "combined_loss"],
            model: ["ColorModel", "apply_model", "generate_lut"],
            scenes: ["extract_dkl_feature", "cluster_dkl_scenes"],
        }
        for module, names in required.items():
            for name in names:
                self.assertTrue(callable(getattr(module, name, None)), name)

    def test_model_features_are_three_normalized_channels(self):
        features = model.model_features(torch.rand(9, 11, 3))
        self.assertEqual(features.shape, (9, 11, 3))
        self.assertGreaterEqual(float(features.min()), 0.0)
        self.assertLessEqual(float(features.max()), 1.0)

    def test_model_chroma_is_zero_for_grayscale(self):
        image = torch.rand(9, 11, 1).expand(-1, -1, 3)
        self.assertTrue(torch.allclose(model.model_features(image)[..., 2], torch.zeros(9, 11)))

    def test_power_target_and_gradient(self):
        original = torch.full((8, 8, 3), 0.5)
        linear_target = color.srgb_to_linear(original) * 0.8
        optimized = color.linear_to_srgb(linear_target).requires_grad_()
        value = losses.power_loss(original, optimized, 0.8)
        self.assertLess(float(value), 1e-10)
        value.backward()
        self.assertTrue(torch.isfinite(optimized.grad).all())

    def test_power_target_is_tilewise(self):
        original = torch.zeros(8, 8, 3)
        original[:4, :4] = 0.2
        original[4:, 4:] = 0.8
        optimized = color.linear_to_srgb(color.srgb_to_linear(original) * 0.8)
        self.assertLess(float(losses.power_loss(original, optimized, 0.8)), 1e-10)

    def test_dkl_clustering(self):
        images = [np.full((4, 4, 3), value, np.float32) for value in (0.1, 0.2, 0.8, 0.9)]
        features = np.stack([scenes.extract_dkl_feature(image) for image in images])
        labels, centers, mean, std = scenes.cluster_dkl_scenes(features, 2, seed=0)
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(centers.shape, (2, 9))
        self.assertTrue(np.isfinite(mean).all() and np.isfinite(std).all())

        manifest = {"centers": centers.tolist(), "mean": mean.tolist(), "std": std.tolist()}
        cluster_id, distance = scenes.match_scene(images[0], manifest, return_distance=True)
        self.assertEqual(cluster_id, scenes.match_scene(images[0], manifest))
        self.assertGreaterEqual(distance, 0.0)

    def test_reporting_power_is_weighted_but_loss_is_not(self):
        red = torch.zeros(4, 4, 3); red[..., 0] = 0.5
        blue = torch.zeros(4, 4, 3); blue[..., 2] = 0.5
        self.assertAlmostEqual(float(power.dynamic_power(red)), float(power.dynamic_power(blue)), places=6)
        self.assertGreater(float(power.display_power(blue)), float(power.display_power(red)))

    def test_all_configs_are_valid(self):
        root = Path(__file__).parents[1] / "configs"
        configs = sorted(root.glob("*.json"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(path=path.name):
                validate_config(json.loads(path.read_text(encoding="utf-8")))

    def test_unknown_config_items_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown config items"):
            validate_config({"loss": {}, "unused_option": True})
        with self.assertRaisesRegex(ValueError, "unknown loss config items"):
            validate_config({"loss": {"unused_loss": 1}})

    def test_pretrain_sampling_caps_each_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in ("a", "b"):
                (root / scene).mkdir()
                for index in range(3):
                    (root / scene / f"{index}.png").touch()
            first = sample_images_per_scene(root, 2, seed=7)
            second = sample_images_per_scene(root, 2, seed=7)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual({path.parent.name for path in first}, {"a", "b"})

    def test_training_images_have_fixed_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, shape in (("wide.png", (320, 180)), ("square.png", (200, 200))):
                path = Path(directory) / name
                Image.new("RGB", shape).save(path)
                paths.append(path)
            self.assertEqual([scenes.load_image(path, 64).shape for path in paths], [(64, 64, 3), (64, 64, 3)])


if __name__ == "__main__":
    unittest.main()
