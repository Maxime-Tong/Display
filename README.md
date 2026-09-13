# Screen Adaptor

A small implementation of learned display-power adaptation with four losses:
ML-PEA-like power targeting, MetaM, Weber, and SSIM. Optional scene-specific
LUTs use the original DKL mean/std/median clustering feature.

See `ALGORITHM.md` for the formulas, compatibility contract, and rebuild path.

## Layout

```text
src/color.py       color conversion
src/power.py       training/evaluation power model
src/losses.py      four losses and combined objective
src/perception/    copied MetaM implementation
src/model.py       multiplicative color model and LUT
src/scenes.py      DKL clustering and manifest
src/pipeline.py    training
eval.py            dataset evaluation
```

## Pretrain, cluster, and fine-tune

Run these commands from the project root, replacing `datasets/train` with the
dataset directory whose immediate subdirectories are scenes:

```powershell
python -m src.pipeline pretrain `
  --data-dir datasets/train `
  --config configs/pretrain_config.json `
  --output-dir outputs

python -m src.pipeline cluster `
  --data-dir datasets/train `
  --config configs/finetune_config.json `
  --manifest outputs/scene_manifest.json

python -m src.pipeline finetune `
  --data-dir datasets/train `
  --config configs/finetune_config.json `
  --base-checkpoint outputs/base_checkpoint.pt `
  --manifest outputs/scene_manifest.json `
  --output-dir outputs
```

Pretraining writes the base checkpoint, base LUT, and training history.
Clustering writes the DKL scene manifest. Fine-tuning writes one checkpoint,
LUT, and history per cluster and updates the manifest LUT paths.

## Evaluate

With `--scene-manifest`, the benchmark-compatible normalized DKL distance threshold defaults to `1.8`. Images above it use the base `--model` or `--lut` and report `cluster_id=-1`; set `--match-distance-threshold 0` to disable switching.

```text
python eval.py --data-dir datasets/test --model outputs/base_checkpoint.pt --output-dir results
python eval.py --data-dir datasets/test --lut outputs/base_lut.pt --output-dir results
python eval.py --data-dir datasets/test --lut outputs/base_lut.pt --scene-manifest outputs/scene_manifest.json --output-dir results
```

Evaluation writes optimized images, `metrics.csv`, and `metrics.json`. Metrics
are ML-PEA-like power saving, PSNR, SSIM, and MetaM. Evaluation automatically
uses CUDA when available; pass `--device cpu` to override it.

## Plot training

Training writes `outputs/training_history.csv` and one history file per DKL
cluster. Render the raw losses and weighted power-saving trend with:

```text
python plot_training.py --history outputs/training_history.csv --output-dir outputs/plots
```

Add `--log-scale` when loss magnitudes differ too much for a linear plot.

## Check

```text
python -m py_compile src/*.py eval.py
python -m unittest discover -s tests
```

Dependencies: Python 3.10+, PyTorch, NumPy, Pillow, scikit-image, and `lpips`.

## Optimize a LUT for 17% saving

Measured results and limitations: [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md).

The offline budget optimizer uses the weighted display-power model reported by
`eval.py` and its exact foveated MetaM settings. It directly learns RGB reduction
allocation at LUT nodes, with 17% weighted linear-power reduction at every node.
It trains through LUT interpolation and selects the lowest validation MetaM.
The legacy network training and its losses remain available unchanged.

```powershell
conda activate 3dgs
python -m src.optimize_lut --data-dir D:/workspace/master/3DGS/Vulkan/display_project/screen_adaptor/datasets
python -m src.optimize_lut --data-dir D:/workspace/master/3DGS/Vulkan/display_project/screen_adaptor/datasets --config configs/lut_optimize_config.json
python -m src.optimize_lut --data-dir D:/workspace/master/3DGS/Vulkan/display_project/screen_adaptor/datasets --evaluate-only --image-size 512
python eval.py --data-dir YOUR_TEST_IMAGES --lut outputs/optimized_lut/base_lut.pt --output-dir results/optimized
```

`eval.py` recursively scans the target directory, evaluates one image at a time,
and preserves subdirectories when `--save` is supplied. Use `--max-images` for
a smoke test; omit it for the full target set. The metrics files contain one
row per image plus a summary, so evaluation does not need the training split.

Defaults: 33³ RGB float32 LUT (421 KiB of tensor data), 400 steps, 256×256
training images, deterministic per-folder sampling of 8 training, 2 validation,
and 3 test images. `split.json` records the paths. These are disjoint image
splits within each dataset folder, not a guarantee of independent video scenes.
For the full 1000×10 corpus, the supplied config assigns 800 images per scene
to training and 200 to validation; it does not reserve an internal held-out
test split.
Training loads one image per step; validation is capped at 200 images, so the
10,000-image corpus is never loaded into RAM.
Nested official dataset partitions (such as LVSQ train/test) are pooled for this
custom sampled experiment; these results are not an official dataset benchmark.
`--evaluate-only` reuses the saved split and LUT, and also compares the old LUT
after calibrating its strength on validation images to the same average saving.

Online work is one trilinear LUT lookup; there is no network, MetaM computation,
power reduction, or scene classification at runtime. The Python implementation
uses native `grid_sample`. A renderer can upload the table as a linear-filtered
3D texture and sample at `(rgb * (N - 1) + 0.5) / N`; use clamp-to-edge and a
linear floating-point texture format, with no automatic sRGB decode. Pack the
RGB array with R as the texture's fastest coordinate (the stored PyTorch array
has B fastest, so transpose spatial axes or reverse lookup coordinates).

The 17% constraint is exact at LUT nodes under the supplied weighted linear-RGB
model. Interpolation and output quantization can change the saving; inspect
the per-image results at the deployed resolution/precision. This optimizes
modeled display dynamic power, not measured total system energy. The runtime
energy cost needs measurement on the target renderer and panel.
