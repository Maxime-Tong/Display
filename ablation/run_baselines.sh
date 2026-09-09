#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
DATASET="/path/to/dataset"
OUTPUT_PATH="results/baselines"
ML_PEA_CHECKPOINT="/path/to/ml-pea/latest.ckpt"
HVS_CHECKPOINT="ablation/hvs_vr_encoding/host/color_optimizer/model/model.pth"
VR_POWER_SAVER_CHECKPOINT="ablation/vr-power-saver/io/color_model/model.pth"

PYTHON="python"
DEVICE="cuda"

mkdir -p "$OUTPUT_PATH"

echo "Running uniform baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline uniform \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/uniform" \
  --device "$DEVICE" \
  --save

echo "Running ML-PEA baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline ml-pea \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/ml_pea" \
  --checkpoint "$ML_PEA_CHECKPOINT" \
  --device "$DEVICE" \
  --save

echo "Running HVS-VR-Encoding baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline hvs-vr-encoding \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/hvs_vr_encoding" \
  --checkpoint "$HVS_CHECKPOINT" \
  --device cpu \
  --save

echo "Running VR-Power-Saver baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline vr-power-saver \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/vr_power_saver" \
  --checkpoint "$VR_POWER_SAVER_CHECKPOINT" \
  --device "$DEVICE" \
  --save

echo "All baseline evaluations finished. Results: $OUTPUT_PATH"
