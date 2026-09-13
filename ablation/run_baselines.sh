#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
DATASET="/data/xthuang/datasets/LVSQ/test"
OUTPUT_PATH="/data/xthuang/workspace/color_display/outputs/LVSQ_metamloss"
ML_PEA_CHECKPOINT="/data/xthuang/workspace/display_optimization/ML-PEA/checkpoints/R0.83/epoch_60.ckpt"
HVS_CHECKPOINT="/data/xthuang/workspace/display_optimization/hvs_vr_encoding/host/color_optimizer/model/model.pth"
VR_POWER_SAVER_CHECKPOINT="/data/xthuang/workspace/display_optimization/vr-power-saver/io/color_model/model.pth"

PYTHON="python"
DEVICE="cuda"

mkdir -p "$OUTPUT_PATH"

echo "Running uniform baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline uniform \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/uniform" \
  --device "$DEVICE"

echo "Running ML-PEA baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline ml-pea \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/ml_pea" \
  --checkpoint "$ML_PEA_CHECKPOINT" \
  --device "$DEVICE"

echo "Running VR-Power-Saver baseline..."
"$PYTHON" ablation/eval_baseline.py \
  --baseline vr-power-saver \
  --data-dir "$DATASET" \
  --output-dir "$OUTPUT_PATH/vr_power_saver" \
  --checkpoint "$VR_POWER_SAVER_CHECKPOINT" \
  --device "$DEVICE"

echo "All baseline evaluations finished. Results: $OUTPUT_PATH"
