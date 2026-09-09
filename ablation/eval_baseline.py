"""Evaluate the four baseline methods with the project's canonical metrics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color import linear_to_srgb, srgb_to_linear
from src.perception import MetamericLoss
from src.power import power_saving
from src.scenes import IMAGE_EXTENSIONS

METRICS = ("power_saving", "psnr", "ssim", "metam", "lpips")
DEFAULT_POWER_WEIGHTS = (0.22970384, 0.24373232, 0.5265638)


def image_paths(data_dir, max_images):
    paths = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:max_images] if max_images else paths


def load_rgb(path, device):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).to(device)


def make_metrics(device):
    import lpips

    metam = MetamericLoss(
        device=device, real_image_width=1.4, real_viewing_distance=0.7,
        equi=False, alpha=5.0, mode="quadratic", loss_type="L1",
        use_l2_foveal_loss=False, n_pyramid_levels=5, n_orientations=4,
        use_radial_weight=True,
    )
    return metam, lpips.LPIPS(net="alex").to(device).eval()


def evaluate_image(original, optimized, metam, lpips_model, power_weights):
    with torch.no_grad():
        original_nchw = original.permute(2, 0, 1)[None]
        optimized_nchw = optimized.permute(2, 0, 1)[None]
        metam_value = metam(optimized_nchw, original_nchw, gaze=[0.5, 0.5])
        lpips_value = lpips_model(optimized_nchw * 2 - 1, original_nchw * 2 - 1).item()
    a, b = original.detach().cpu().numpy(), optimized.detach().cpu().numpy()
    return {
        "power_saving": float(power_saving(original, optimized, power_weights)),
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
        "metam": float(metam_value),
        "lpips": float(lpips_value),
    }


class Uniform:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        return linear_to_srgb(self.alpha * srgb_to_linear(image))


class MLPEA:
    def __init__(self, checkpoint, method, channels, device):
        path = ROOT / "ablation" / "ML-PEA" / "src"
        sys.path.insert(0, str(path))
        from unet import UNet
        self.method = method
        self.model = UNet(3, channels).to(device)
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state.get("state_dict", state))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image):
        mask = self.model(image.permute(2, 0, 1)[None])
        if self.method == "MULT":
            output = (mask + 1) / 2 * image.permute(2, 0, 1)[None]
        else:
            output = image.permute(2, 0, 1)[None] - (mask + 1) / 2
        return output[0].permute(1, 2, 0).clamp(0, 1)


class HVS:
    def __init__(self, checkpoint, device, fov, max_ecc, tile_size, ecc_no_compress, abc_scaler):
        if device.type != "cpu":
            raise ValueError("hvs-vr-encoding uses its CPU implementation; pass --device cpu")
        path = ROOT / "ablation" / "hvs_vr_encoding" / "host" / "color_optimizer"
        sys.path.insert(0, str(path))
        from red_blue_optimization_cpu import Image_color_optimizer
        self.checkpoint = checkpoint
        self.fov, self.max_ecc = fov, max_ecc
        self.tile_size = tile_size
        self.ecc_no_compress, self.abc_scaler = ecc_no_compress, abc_scaler
        self.ImageColorOptimizer = Image_color_optimizer

    def __call__(self, image):
        original = image.detach().cpu().numpy()
        h, w = original.shape[:2]
        pad_h, pad_w = (-h) % self.tile_size, (-w) % self.tile_size
        padded = np.pad(original, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        optimizer = self.ImageColorOptimizer(
            foveated=True, max_ecc=self.max_ecc, h_fov=self.fov,
            img_height=padded.shape[0], img_width=padded.shape[1],
            tile_size=self.tile_size, abc_scaler=self.abc_scaler,
            ecc_no_compress=self.ecc_no_compress,
        )
        optimizer.Tile_color_optimizer.color_model.load(self.checkpoint)
        optimizer.Tile_color_optimizer.color_model.to_eval()
        output = optimizer.color_conversion((padded * 255).astype(np.float32)) / 255.0
        return torch.from_numpy(output[:h, :w]).to(image.device, dtype=image.dtype)


class VRPowerSaver:
    def __init__(self, checkpoint, fov, transition_width):
        path = ROOT / "ablation" / "vr-power-saver"
        sys.path.insert(0, str(path))
        from color_model.base_color_model import BaseColorModel
        from util.vr_tools import build_ecc_map, build_transition_mask
        # The vr-power-saver invoker treats a supplied mapping as the complete
        # config, so pass its model defaults explicitly (the original demo does
        # this through the module argument parser).
        self.model = BaseColorModel(dict(BaseColorModel.args()))
        self.model.load(checkpoint)
        self.build_ecc_map = build_ecc_map
        self.build_transition_mask = build_transition_mask
        self.fov, self.transition_width = fov, transition_width

    def __call__(self, image):
        inp = image.detach().cpu().numpy()
        h, w = inp.shape[:2]
        ecc = self.build_ecc_map(self.fov, 0.0, 0.0, self.model.opt.max_eccentricity, h, w)
        power_vec = -np.array([231.5384684, 245.6795914, 530.7596369])
        output = self.model.apply_filter(inp, ecc, power_vec)
        mask = self.build_transition_mask(ecc, self.model.opt.min_eccentricity, self.transition_width)
        output = inp * (1 - mask) + output * mask
        return torch.from_numpy(output).to(image.device, dtype=image.dtype).clamp(0, 1)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a baseline with canonical display-project metrics")
    parser.add_argument("--baseline", required=True, choices=("uniform", "ml-pea", "hvs-vr-encoding", "vr-power-saver"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--power-weights", nargs=3, type=float, default=DEFAULT_POWER_WEIGHTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-alpha", type=float, default=0.83)
    parser.add_argument("--checkpoint")
    parser.add_argument("--method", choices=("MULT", "ADD"), default="MULT")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--transition-width", type=float, default=3.0)
    parser.add_argument("--max-ecc", type=float, default=30.0)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--ecc-no-compress", type=float, default=10.0)
    parser.add_argument("--abc-scaler", type=float, default=1.0)
    parser.add_argument("--save", action="store_true")
    return parser


def make_inference(args, device):
    if args.baseline == "uniform":
        if not 0 < args.target_alpha <= 1:
            raise ValueError("--target-alpha must be in (0, 1]")
        return Uniform(args.target_alpha)
    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required for {args.baseline}")
    if args.baseline == "ml-pea":
        return MLPEA(args.checkpoint, args.method, args.channels, device)
    if args.baseline == "hvs-vr-encoding":
        return HVS(args.checkpoint, device, args.fov, args.max_ecc, args.tile_size, args.ecc_no_compress, args.abc_scaler)
    return VRPowerSaver(args.checkpoint, args.fov, args.transition_width)


def main():
    args = build_parser().parse_args()
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = image_paths(args.data_dir, args.max_images)
    if not paths:
        raise ValueError("dataset contains no images")
    inference = make_inference(args, device)
    metam, lpips_model = make_metrics(device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for current, path in enumerate(paths, 1):
        original = load_rgb(path, device)
        optimized = inference(original)
        row = {"filename": str(path.relative_to(args.data_dir)), **evaluate_image(original, optimized, metam, lpips_model, args.power_weights)}
        rows.append(row)
        if args.save:
            pixels = (optimized.detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(output / path.name)
        print(f"[{current}/{len(paths)}] {path.name} " + " ".join(f"{key}={row[key]:.6f}" for key in METRICS))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    config = {key: value for key, value in vars(args).items() if key not in {"save", "device"}}
    (output / "metrics.json").write_text(json.dumps({"config": config, "images": rows, "summary": summary}, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print("summary", summary)


if __name__ == "__main__":
    main()
