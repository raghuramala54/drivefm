#!/usr/bin/env bash
# Full experiment grid for the paper: 6 models x 3 seeds, ablations, ensembles, aggregation.
# Usage:  bash scripts/run_experiments.sh  [TRAIN_CSV] [TEST_CSV]
set -euo pipefail
TRAIN=${1:-data/track1/driving_log.csv}
TEST=${2:-data/track2_eval/driving_log.csv}
SEEDS=${SEEDS:-"0 1 2"}
COMMON="data.train_logs=[$TRAIN] data.test_logs=[$TEST]"

# --- main comparison
for cfg in pilotnet_mse resnet18_mse dinov2_mse dinov2_mdn dinov2_flow dinov2_flow_ft; do
  for s in $SEEDS; do
    [ -f runs/${cfg}_s${s}/metrics.json ] && continue
    python scripts/train.py --config configs/$cfg.yaml --seed $s --set $COMMON
  done
done

# --- ablations of the proposed method (one factor at a time)
abl() { # name, overrides...
  local name=$1; shift
  for s in $SEEDS; do
    [ -f runs/${name}_s${s}/metrics.json ] && continue
    python scripts/train.py --config configs/dinov2_flow.yaml --seed $s --set $COMMON name=$name "$@"
  done
}
abl abl_flow_nochunk     data.chunk=1
abl abl_flow_noctx       data.context_frames=1 model.temporal_layers=0
abl abl_flow_speedinput  model.use_speed=true
abl abl_flow_meandecode  model.flow_decode=mean
abl abl_flow_kdemode     model.flow_decode=kde_mode   # single-sample mode (no mean-shift)
abl abl_flow_attnpool    model.pool=attn             # learned-query attention pooling
abl abl_flow_randomsplit data.split_mode=random   # leaky split: compare val-vs-test gap

# --- deep ensembles (epistemic uncertainty) on the held-out track
python scripts/evaluate.py --ckpt runs/pilotnet_mse_s*/best.pt --logs $TEST --out runs/ensemble_pilotnet_mse
python scripts/evaluate.py --ckpt runs/dinov2_flow_s*/best.pt  --logs $TEST --out runs/ensemble_dinov2_flow
# MC-dropout baseline for the deterministic head
python scripts/evaluate.py --ckpt runs/dinov2_mse_s0/best.pt --logs $TEST --mc_dropout --out runs/mcdrop_dinov2_mse

# --- tables
python scripts/aggregate.py --runs runs --split test --out runs/table_test.tex
python scripts/aggregate.py --runs runs --split val
