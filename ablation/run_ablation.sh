#!/usr/bin/env bash
set -euo pipefail

DATA="" OUT="" MODEL="" MAX=0 DEVICE="" RECIPES="" RUNTIME=0 BLOCKS="" LUTRES=0 CLUSTERS="1,2,4,8" RESOLUTIONS="8,16,32"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--data-dir) DATA="$2"; shift 2;; -o|--out) OUT="$2"; shift 2;; -m|--model-dir) MODEL="$2"; shift 2;;
    --max-images) MAX="$2"; shift 2;; --device) DEVICE="$2"; shift 2;; --ablation-recipe) RECIPES="$2"; shift 2;;
    --ablation-runtime) RUNTIME=1; shift;; --sens-block) BLOCKS="$2"; shift 2;; --sens-lutres) LUTRES=1; shift;;
    --clusters) CLUSTERS="$2"; shift 2;; --lut-resolutions) RESOLUTIONS="$2"; shift 2;; *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$DATA" && -n "$OUT" ]] || { echo "--data-dir and --out are required" >&2; exit 2; }
DEV=(); [[ -n "$DEVICE" ]] && DEV=(--device "$DEVICE")
IFS=',' read -ra KS <<< "$CLUSTERS"

if [[ -n "$RECIPES" ]]; then
  IFS=',' read -ra RS <<< "$RECIPES"
  for recipe in "${RS[@]}"; do
    dir="$OUT/recipes/$recipe"; python run_recipes.py --data-dir "$DATA" --output-dir "$dir" --recipe "$recipe" --clusters "${KS[${#KS[@]}-1]}"
    mode=cluster; [[ "$recipe" == pretrain ]] && mode=single
    python run_eval.py --model-dir "$dir" --data-dir "$DATA" --out "$OUT/eval/recipe/$recipe.json" --mode "$mode" --max-images "$MAX" "${DEV[@]}"
  done
fi

if [[ "$RUNTIME" == 1 ]]; then
  [[ -n "$MODEL" ]] || { echo "--model-dir is required for runtime ablation" >&2; exit 2; }
  python run_eval.py --model-dir "$MODEL" --data-dir "$DATA" --out "$OUT/eval/runtime/lut_bs4.json" --repr lut --block-size 4 --max-images "$MAX" "${DEV[@]}"
  python run_eval.py --model-dir "$MODEL" --data-dir "$DATA" --out "$OUT/eval/runtime/net.json" --repr net --block-size 1 --max-images "$MAX" "${DEV[@]}"
  python run_eval.py --model-dir "$MODEL" --data-dir "$DATA" --out "$OUT/eval/runtime/lut_bs1.json" --repr lut --block-size 1 --max-images "$MAX" "${DEV[@]}"
fi

if [[ -n "$BLOCKS" ]]; then
  [[ -n "$MODEL" ]] || { echo "--model-dir is required for block sensitivity" >&2; exit 2; }
  IFS=',' read -ra BS <<< "$BLOCKS"; for block in "${BS[@]}"; do
    python run_eval.py --model-dir "$MODEL" --data-dir "$DATA" --out "$OUT/eval/block/bs$block.json" --block-size "$block" --max-images "$MAX" "${DEV[@]}"
  done
fi

if [[ "$LUTRES" == 1 ]]; then
  IFS=',' read -ra LS <<< "$RESOLUTIONS"; for k in "${KS[@]}"; do
    dir="$OUT/lutres/k$k"; python run_recipes.py --data-dir "$DATA" --output-dir "$dir" --recipe pretrain_finetune --clusters "$k"
    for res in "${LS[@]}"; do python run_eval.py --model-dir "$dir" --data-dir "$DATA" --out "$OUT/eval/lutres/k${k}_r${res}.json" --lut-resolution "$res" --max-images "$MAX" "${DEV[@]}"; done
  done
fi

python run_eval.py --report --results-dir "$OUT/eval"
