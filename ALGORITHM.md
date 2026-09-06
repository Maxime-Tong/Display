# Screen Adaptor Algorithm Rebuild

This specification defines a clean rebuild of `screen_adaptor`. It keeps the
existing algorithmic functions while separating responsibilities for review
and maintenance.

## Scope

Keep multiplicative per-pixel color adaptation, an ML-PEA-like global power
target loss, MetaM, Weber, SSIM, original DKL scene clustering, base training,
optional per-cluster fine-tuning, and LUT export. Do not add new clustering
methods or optimization strategies.

## Minimal layout

```text
screen_adaptor/
├── .gitignore
├── README.md
├── configs/
├── src/
│   ├── color.py       # sRGB/linear and DKL conversions
│   ├── power.py       # display-power measurement
│   ├── losses.py      # power, Weber, SSIM, and combined loss
│   ├── perception/    # copied directly from the existing implementation
│   ├── model.py       # color network and LUT generation
│   ├── scenes.py      # original DKL clustering
│   └── pipeline.py    # training/inference orchestration
├── tests/
└── eval.py
```

`pipeline.py` orchestrates. It must not duplicate formulas owned by the other
modules. The existing `perception/` directory is copied as a self-contained
package first; do not rewrite its MetaM implementation during this rebuild.

The package is imported directly from `src`; do not create a nested
`src/screen_adaptor/` package in the rebuilt folder.

## Algorithm

1. Load images as sRGB float tensors in `[0, 1]`.
2. Predict a multiplicative gain per pixel and produce:

   ```python
   gain = (tanh(head(features)) + 1) / 2
   optimized = clamp(input * gain, 0, 1)
   ```

3. Convert input and optimized images to linear RGB.
4. Compute the effective ML-PEA dynamic power:

   ```python
   input_power = mean(input_linear[..., 0]) \\
               + mean(input_linear[..., 1]) \\
               + mean(input_linear[..., 2])
   output_power = mean(output_linear[..., 0]) \\
                + mean(output_linear[..., 1]) \\
                + mean(output_linear[..., 2])
   target_power = target_alpha * input_power
   power_loss = (output_power - target_power) ** 2
   ```

   This is like ML-PEA: global power target plus squared error. `target_alpha`
   is the retained-power factor; the original ML-PEA setup uses `r = 0.8`.

5. Compute the other three losses independently:

   ```python
   metam_loss = metameric_loss(optimized, input)
   weber_loss = mean(weber_relative_error(
       input, optimized, weights=power_weights, eps=weber_epsilon
   ))
   ssim_loss = 1 - ssim(input, optimized)
   ```

6. Combine them once:

   ```python
   total_loss = (
       lambda_power * power_loss
       + lambda_metam * metam_loss
       + lambda_weber * weber_loss
       + lambda_ssim * ssim_loss
   )
   ```

Return raw components and `total_loss` for logging. Apply each weight exactly
once, in the combined-loss function.

## Mathematical formulation

Let an input image be (I \in [0,1]^{H \times W \times 3}) in sRGB space. Let
(γ^{-1}) be the sRGB-to-linear transfer function:

\[
I_{lin}(x,c) =
\begin{cases}
I(x,c)/12.92, & I(x,c) \le 0.04045 \\
((I(x,c)+0.055)/1.055)^{2.4}, & \text{otherwise.}
\end{cases}
\]

The model predicts a gain (G(x,c) \in [0,1]). The optimized image is:

\[
O(x,c) = \operatorname{clip}(I(x,c)G(x,c), 0, 1).
\]

Define the effective ML-PEA-like dynamic power as the sum of channel means:

\[
P(X) = \frac{1}{HW}\sum_x X_{lin}(x,R)
      + \frac{1}{HW}\sum_x X_{lin}(x,G)
      + \frac{1}{HW}\sum_x X_{lin}(x,B).
\]

The target power and power loss are:

\[
P_{target} = \alpha P(I),
\qquad
L_{power} = (P(O)-P_{target})^2.
\]

The MetaM, Weber, and SSIM losses are:

\[
L_{MetaM} = \operatorname{MetaM}(O,I),
\]

\[
L_{Weber} = \frac{1}{3HW}\sum_{x,c}
w_c\left|\frac{O(x,c)-I(x,c)}{I(x,c)+\epsilon}\right|,
\]

\[
L_{SSIM} = 1-\operatorname{SSIM}(I,O).
\]

The final objective is:

\[
L = \lambda_{power}L_{power}
  + \lambda_{MetaM}L_{MetaM}
  + \lambda_{Weber}L_{Weber}
  + \lambda_{SSIM}L_{SSIM}.
\]

The exact implementation of `weber_relative_error` remains authoritative for
the Weber term; the equation above describes its intended weighted relative
error behavior.

## Training and inference

Train one base model and generate its LUT. Optionally extract the existing DKL
scene feature, cluster scenes with the original DKL clustering function only,
fine-tune one model/LUT per cluster, and save a manifest mapping scenes to
clusters. If clustering is disabled, use the base LUT.

## Evaluation script

Provide one top-level `eval.py` entry point. It loads a dataset, loads either a
base model/LUT or a selected cluster LUT, writes optimized images, and reports
one row per image plus a dataset summary.

Example:

```text
python eval.py \\
  --data-dir datasets/example \\
  --model outputs/base_checkpoint.pt \\
  --lut outputs/base_lut.pt \\
  --output-dir results/example
```

The evaluation path is:

```python
original = load_image(path)                 # sRGB float HWC [0, 1]
optimized = apply_model_or_lut(model, original)
save_image(output_path, optimized)
metrics = evaluate_image(original, optimized)
```

Required metrics:

```python
power_saving = 1 - dynamic_power(optimized) / dynamic_power(original)
psnr = PSNR(original, optimized, data_range=1.0)
ssim = SSIM(original, optimized, data_range=1.0)
metam = MetaM(optimized, original)
```

Power saving must use the same ML-PEA-like dynamic-power function as training:

```python
dynamic_power(image) = mean(R_linear) \\
                     + mean(G_linear) \\
                     + mean(B_linear)
```

For zero-power inputs, return `power_saving = 0.0` instead of dividing by zero.
Write `metrics.json`, `metrics.csv`, and a short summary to stdout. Do not
silently use a different power model in evaluation.

## Public functions

```text
color.py:  srgb_to_linear, linear_to_srgb, rgb_to_dkl
power.py:  dynamic_power, target_power, power_saving
losses.py: power_loss, metameric_loss, weber_loss, ssim_loss, combined_loss
model.py:  ColorModel, apply_model, generate_lut
scenes.py: extract_dkl_feature, cluster_dkl_scenes, load_scene_manifest,
           save_scene_manifest
```

Keep existing names where callers require compatibility. Each function should
have one responsibility and document its input/output shape.

## Compatibility contract for future rebuilds

Treat this section as an interface contract. A future implementation may move
code between files, but must preserve these callable names and meanings unless
the contract is deliberately versioned:

```python
# color.py
def srgb_to_linear(image): ...       # (..., 3) -> (..., 3), float [0, 1]
def linear_to_srgb(image): ...       # (..., 3) -> (..., 3), float [0, 1]
def rgb_to_dkl(image): ...           # (..., 3) -> (..., 3)

# power.py
def dynamic_power(image): ...        # (..., 3) -> scalar tensor/float
def target_power(image, alpha): ...  # (..., 3) -> scalar tensor/float
def power_saving(original, optimized): ...

# losses.py
def power_loss(original, optimized, target_alpha): ...
def metameric_loss(original, optimized): ...
def weber_loss(original, optimized, weights, epsilon): ...
def ssim_loss(original, optimized): ...
def combined_loss(original, optimized, config): ...

# scenes.py
def extract_dkl_feature(image): ...
def cluster_dkl_scenes(features, number_of_clusters): ...

# model.py
def apply_model(model, image): ...
def generate_lut(model, resolution, device): ...
```

The rebuild must preserve:

1. input/output channel order: RGB;
2. image range: sRGB floats in `[0, 1]`;
3. image layout at public boundaries: `HWC` unless a function explicitly says
   `NCHW`;
4. differentiability of PyTorch training functions;
5. `power_loss` as the raw squared loss, with its weight applied only by
   `combined_loss`;
6. DKL as the only scene-clustering method;
7. base-LUT and cluster-LUT output compatibility.

## Future-chat build instructions

Start a new implementation chat by attaching or quoting this document and
giving the instruction:

```text
Use screen_adaptor_algorithm_refactor.md as the source of truth.
Rebuild the project incrementally. Preserve every function in the
Compatibility contract, including names, signatures, shapes, ranges,
mathematical behavior, and differentiability. Do not add algorithms outside
the Scope section. Before editing, inventory existing callers. After each
module, run the smallest relevant test and report any intentional API change.
```

The document is the durable handoff; a new chat cannot reliably infer these
requirements from conversation history alone.

## Rebuild path

Follow this order so every step remains reviewable:

```text
1. Create the new folder and copy only the required dataset/config examples.
2. Copy the existing perception/ directory directly into src/perception/ and
   verify its MetaM import and output on one image.
3. Create src/color.py and verify round-trip sRGB conversion.
4. Create src/power.py and verify the ML-PEA-like power equation.
5. Create src/losses.py and verify power, Weber, SSIM, and combined loss.
6. Create src/model.py and verify one image can produce a valid output.
7. Create src/scenes.py using only the existing DKL feature and clustering code.
8. Create src/pipeline.py for training and LUT export.
9. Create eval.py and verify metrics on one image.
10. Run the full small-dataset smoke test.
```

Recommended commands from the rebuilt project root:

```text
python -m py_compile src/*.py eval.py
python -m unittest discover -s tests
python eval.py --data-dir datasets/smoke --model outputs/base_checkpoint.pt --output-dir results/smoke
```

Do not begin with a broad rewrite. After each step, compare the new function
against its old counterpart using the same small input and record any numeric
difference.

## Exact source-to-function mapping

Use this mapping while rebuilding; it prevents accidental loss of behavior:

```text
old color_ops.py       -> src/color.py
old power.py           -> src/power.py
old perception/       -> src/perception/ (copy directly)
old pipeline.py losses -> src/losses.py, excluding MetaM implementation
old model.py           -> src/model.py
old scene_matcher.py   -> src/scenes.py
old pipeline.py train  -> src/pipeline.py
benchmark metric code  -> eval.py
```

Copy the function body first, run the contract tests, then simplify only when
the outputs and gradients match. Keep compatibility wrappers temporarily if
an existing caller still uses an old name.

## Rebuild invariants

The following assertions must remain true:

```python
assert image.dtype is floating_point
assert image.shape[-1] == 3
assert image.min() >= 0 and image.max() <= 1
assert dynamic_power(image) >= 0
assert power_loss(image, image, alpha) >= 0
assert combined_loss(image, image, config).isfinite()
```

For a multiplicative gain of `alpha` in linear RGB, the output power should be
approximately `alpha * input_power`. For an output whose power is exactly the
target power, `power_loss` must be zero up to floating-point tolerance.

## Minimal contract test

Keep one small executable test that fails if the public contract changes:

```python
def test_public_contract():
    import inspect
    from screen_adaptor import color, losses, model, power, scenes

    required = {
        color: ["srgb_to_linear", "linear_to_srgb", "rgb_to_dkl"],
        power: ["dynamic_power", "target_power", "power_saving"],
        losses: ["power_loss", "metameric_loss", "weber_loss", "ssim_loss", "combined_loss"],
        model: ["apply_model", "generate_lut"],
        scenes: ["extract_dkl_feature", "cluster_dkl_scenes"],
    }
    for module, names in required.items():
        for name in names:
            assert callable(getattr(module, name, None)), name
```

Add numerical tests for the power equation, zero loss at the target power, and
finite gradients. Do not require a large test framework for this contract.

## Simple `.gitignore`

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
env/
*.pt
*.pth
*.ckpt
outputs/
results/
logs/
*.log
.idea/
.vscode/
Thumbs.db
```

## Review criteria

- Power uses global means of linear RGB channels and a squared target error.
- MetaM, Weber, SSIM, and power are independently callable.
- Loss weights are applied once.
- DKL is the only scene-clustering algorithm.
- Base and cluster LUTs share one model interface.
- Training orchestration contains no duplicated loss or color conversion code.
