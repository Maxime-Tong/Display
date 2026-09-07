# Screen Adaptor

Learn a lightweight scalar power factor from three per-pixel features:

- linear luminance;
- local texture strength from finite differences;
- normalized radial screen position.

The network applies the factor directly to linear RGB with a fixed per-channel
compensation vector. Its output activation is bounded near identity:

```text
factor = 1 - max_attenuation * sigmoid(network(features))
```

The final network layer starts near factor `1`. After training, the mapping is
exported as a trilinearly interpolated `16 x 16 x 16` scalar LUT, so deployment
only needs feature extraction and one LUT lookup per pixel.

## Train

```powershell
python -m src.pipeline --data-dir datasets/train --config configs/train_config.json --output-dir outputs
```

Training writes `factor_checkpoint.pt`, `factor_lut.pt`, and
`training_history.csv`.

## Evaluate

```powershell
python eval.py --data-dir datasets/test --lut outputs/factor_lut.pt --output-dir results
python eval.py --data-dir datasets/test --model outputs/factor_checkpoint.pt --output-dir results
```

Add `--save` to write optimized images. Evaluation reports power saving, PSNR,
SSIM, and MetaM.

## Check

```powershell
python -m py_compile src/*.py eval.py
python -m unittest discover -s tests
```
