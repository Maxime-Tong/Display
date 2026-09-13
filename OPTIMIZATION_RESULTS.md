# LUT optimization results — 2026-09-13

The exported `outputs/optimized_lut/base_lut.pt` achieves **17.028% mean modeled
display-power saving** and **18.02% lower MetaM than uniform linear dimming** on
30 sampled test images at their native resolutions. The original root LUT was
left untouched.

| Native-resolution result | Uniform dimming LUT | Optimized LUT |
| --- | ---: | ---: |
| Mean weighted dynamic-power saving | 16.9994% | 17.0281% |
| Mean MetaM, lower is better | 0.00710551 | 0.00582534 |

The optimized LUT's native-resolution float savings range from 17.0079% to
17.0682%. After rounding output to 8-bit RGB, mean saving is 17.0218%, with a
range of **16.9972–17.0578%**. Therefore this is an average 17% result, not a
strict per-frame minimum guarantee after quantization.

At 256×256, MetaM decreased from 0.00736686 to 0.00565113 (23.29%). At 512×512,
it decreased from 0.00738222 to 0.00576884 (21.85%). Native-resolution results
above are the primary quality comparison; resizing changes the metric.

## Changes

- Directly optimize a 33³ RGB LUT against the same foveated MetaM configuration
  used by `eval.py`. At each LUT node, allocate a fixed 17% weighted linear-RGB
  power reduction between channels, with no channel brightening. Train through
  interpolation so the optimized representation is the deployed representation.
- Keep all optimization offline. Runtime uses only a LUT lookup, with no neural
  network, scene classification, power measurement, or perceptual loss.
- Replace eight Python corner-gather passes with native 3D `grid_sample`.
  Legacy network training and its compatibility contract remain available.

The power weights are `(0.22970384, 0.24373232, 0.5265638)`. The original training
loss targets unweighted RGB power and uses uniform MetaM, while evaluation
reports weighted power and foveated MetaM. The new route uses the evaluation
definitions consistently without changing legacy loss behavior.

## Runtime

On the RTX 4070 Laptop GPU in the `3dgs` environment, a 1920×1080 RGB float32
frame took a median **4.65 ms**, versus **29.70 ms** for the previous lookup:
**6.38× faster**. Timings use 10 warmups and 30 CUDA-event measurements, without
training running concurrently. Maximum output difference was 2.98e-7.

The table contains 431,244 bytes (421.1 KiB) of RGB float32 data. Production
integration should use a hardware-filtered 3D texture, ideally within the
existing display pass; texture packing and coordinates are documented in README.
GPU timings are not energy measurements. Net system power and panel-specific
savings still require measurements on the target hardware.

## Experiment and limits

Seed 17; 80 training, 20 validation, 30 test images, sampled evenly across all
ten immediate dataset folders. Train at 256×256 for 400 Adam steps, learning
rate 0.03. The lowest validation MetaM selected step 300; test images were not
used for checkpoint selection. The same selected LUT was checked at 512×512
and native resolution without retraining.

This is a custom image-disjoint split. Neighboring frames may be correlated;
nested official partitions, including LVSQ train/test, were pooled. It does
not establish unseen-scene generalization or an official dataset benchmark.
The learned solution improves the measured objective; it is not a proof of
the globally minimum MetaM over all possible LUT algorithms.

The supplied root `base_lut.pt` contains values from 0.814 to 0.981, including
nonzero output for black. It violates this repository's multiplicative
output-color LUT contract. Its raw diagnostic results in `metrics.json` are
excluded from meaningful quality comparisons. The comparison above uses a
valid uniform-dimming LUT at the same nominal power target.

## Reproduce and use

See README for training and evaluation commands. Load the exported artifact
with the existing `eval.py --lut outputs/optimized_lut/base_lut.pt` interface.
The output directory includes:

- `config.json`, `split.json`, `history.json`: parameters, sampled paths, validation history.
- `metrics.json`, `metrics_512.json`, `metrics_native.json`: per-image quality results.
- `runtime.json`: timings and native-resolution float/8-bit power results.
- `verify_runtime.py`, `verify_native_metam.py`: scripts runnable from the project root.

Validation: all **12 unit tests pass**, including node power budgets, finite
gradients, output bounds, black preservation, native interpolation values and
gradients, and offline calibration. `git diff --check` passes.
