#!/usr/bin/env bash
# DriveLite benchmark (no external simulator needed). Generalisation matrix:
#   rows = models, columns = {unseen track shapes (meadow), unseen shapes + unseen style (snow, dusk)}
# Usage: bash scripts/run_drivelite_benchmark.sh        (env vars below override the defaults)
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_TRACKS=${TRAIN_TRACKS:-0-59}
TEST_TRACKS=${TEST_TRACKS:-1000-1019}
MODELS=${MODELS:-"pilotnet_mse resnet18_mse dinov2_mse dinov2_mdn dinov2_flow dinov2_flow_ft"}
SIM="configs/sims/drivelite.yaml"
CL="results/closed_loop_drivelite"

# ---- 1. data: DART expert demonstrations on style 0, held-out offline test sets
[ -f data/dl_train/driving_log.csv ]      || python scripts/collect.py --env drivelite --out data/dl_train --seeds $TRAIN_TRACKS --style 0 --noise 0.15
[ -f data/dl_test_shape/driving_log.csv ] || python scripts/collect.py --env drivelite --out data/dl_test_shape --seeds 1000-1009 --style 0 --noise 0
[ -f data/dl_test_snow/driving_log.csv ]  || python scripts/collect.py --env drivelite --out data/dl_test_snow --seeds 1000-1009 --style 2 --noise 0

# ---- 2. training grid
for m in $MODELS; do for s in $SEEDS; do
  [ -f runs/dl_${m}_s$s/metrics.json ] || python scripts/train.py --config configs/$m.yaml $SIM --seed $s --out runs/dl_${m}_s$s --set name=dl_$m
done; done

# ---- 3. ablations of the proposed model (one factor at a time)
abl() { local name=$1; shift; for s in $SEEDS; do
  [ -f runs/dl_${name}_s$s/metrics.json ] || python scripts/train.py --config configs/dinov2_flow.yaml $SIM --seed $s --out runs/dl_${name}_s$s --set name=dl_$name "$@"
done; }
abl abl_nochunk    data.chunk=1
abl abl_noctx      data.context_frames=1 model.temporal_layers=0
abl abl_speedin    model.use_speed=true
abl abl_meandec    model.flow_decode=mean
abl abl_attnpool   model.pool=attn
abl abl_randsplit  data.split_mode=random

# ---- 4. closed loop on unseen tracks: same style / unseen styles; speed gate on (+ off for the main model)
for d in runs/dl_*_s*/; do r=$(basename $d)
  for st in "0 shape" "2 snow" "3 dusk"; do set -- $st
    ls $CL/$r/*_$2/summary.json >/dev/null 2>&1 && continue
    python scripts/closed_loop_eval.py --env drivelite --style $1 --seeds $TEST_TRACKS --ckpt $d/best.pt --out $CL/$r --tag $2
  done
done
# gate ablation for the main model: no gate / sigma only / novelty only (default = max of both)
for s in $SEEDS; do for st in "0 shape" "2 snow"; do set -- $st
  for g in "--no_gate" "--unc_source sigma" "--unc_source ood"; do
    python scripts/closed_loop_eval.py --env drivelite --style $1 --seeds $TEST_TRACKS --ckpt runs/dl_dinov2_flow_s$s/best.pt --out $CL/dl_dinov2_flow_s$s --tag $2 $g
  done
done; done
# deep ensembles
for m in pilotnet_mse dinov2_flow; do for st in "0 shape" "2 snow"; do set -- $st
  python scripts/closed_loop_eval.py --env drivelite --style $1 --seeds $TEST_TRACKS --ckpt runs/dl_${m}_s*/best.pt --out $CL/ens_$m --tag $2
done; done

# ---- 5. DAgger: classic vs uncertainty-gated queries (label efficiency)
for q in all uncertainty; do
  [ -f runs/dl_dagger_$q/dagger_history.json ] || python scripts/dagger.py --env drivelite --style 0 \
     --init_ckpt runs/dl_dinov2_flow_s0/best.pt --base_logs data/dl_train/driving_log.csv \
     --iters 3 --seeds_per_iter 10 --query $q --eval_seeds $TEST_TRACKS --out runs/dl_dagger_$q
done

# ---- 6. tables
python scripts/aggregate.py --runs runs --split test --dataset dl_test_shape --out results/dl_offline_shape.tex
python scripts/aggregate.py --runs runs --split test --dataset dl_test_snow  --out results/dl_offline_snow.tex
python scripts/aggregate_closed_loop.py --dir "$CL/*" --out results/dl_closed_loop.tex
