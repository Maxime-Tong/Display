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

```text
python eval.py --data-dir datasets/test --model outputs/base_checkpoint.pt --output-dir results
python eval.py --data-dir datasets/test --lut outputs/base_lut.pt --output-dir results
python eval.py --data-dir datasets/test --lut outputs/base_lut.pt --scene-manifest outputs/scene_manifest.json --output-dir results
```

Evaluation writes optimized images, `metrics.csv`, and `metrics.json`. Metrics
are ML-PEA-like power saving, PSNR, SSIM, and MetaM.

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

Dependencies: Python 3.10+, PyTorch, NumPy, Pillow, and scikit-image.
