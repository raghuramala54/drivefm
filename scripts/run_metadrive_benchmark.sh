#!/usr/bin/env bash
# MetaDrive benchmark: train on procedurally generated maps 0-99, test on unseen maps 1000+.
# Needs Python 3.9-3.11 + `pip install -r requirements-metadrive.txt` and a GPU (camera rendering).
# On a headless server: export PREFIX="xvfb-run -a -s '-screen 0 1024x768x24'"
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_MAPS=${TRAIN_MAPS:-0-99}
TEST_MAPS=${TEST_MAPS:-1000-1019}
MODELS=${MODELS:-"pilotnet_mse resnet18_mse dinov2_mse dinov2_flow"}
PREFIX=${PREFIX:-}
SIM="configs/sims/metadrive.yaml"
CL="results/closed_loop_metadrive"

[ -f data/md_train/driving_log.csv ]     || eval $PREFIX python scripts/collect.py --env metadrive --out data/md_train --seeds $TRAIN_MAPS --noise 0.15
[ -f data/md_test_maps/driving_log.csv ] || eval $PREFIX python scripts/collect.py --env metadrive --out data/md_test_maps --seeds 1000-1009 --noise 0

for m in $MODELS; do for s in $SEEDS; do
  [ -f runs/md_${m}_s$s/metrics.json ] || python scripts/train.py --config configs/$m.yaml $SIM --seed $s --out runs/md_${m}_s$s --set name=md_$m
  eval $PREFIX python scripts/closed_loop_eval.py --env metadrive --seeds $TEST_MAPS --ckpt runs/md_${m}_s$s/best.pt --out $CL/md_${m}_s$s --tag maps
  # harder: unseen maps WITH traffic (crashes count as failures)
  eval $PREFIX python scripts/closed_loop_eval.py --env metadrive --traffic 0.1 --seeds $TEST_MAPS --ckpt runs/md_${m}_s$s/best.pt --out $CL/md_${m}_s$s --tag maps_traffic
done; done

python scripts/aggregate.py --runs runs --split test --dataset md_test_maps --out results/md_offline.tex
python scripts/aggregate_closed_loop.py --dir "$CL/*" --out results/md_closed_loop.tex
