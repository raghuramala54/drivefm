#!/usr/bin/env bash
# Focused study: "Knowing When You Don't Know: Uncertainty-Aware End-to-End Driving Under Visual Domain Shift"
#
#   2x2 models  = {PilotNet from scratch, frozen DINOv2} x {MSE head, flow-matching head}, 3 seeds each
#   train       = DriveLite style 0 (meadow) tracks
#   test        = closed loop on UNSEEN tracks in 4 styles: meadow (shape shift only), desert, snow, dusk
#   uncertainty = speed gate off / sigma (action spread) / ood (feature novelty) / max(both); 3-seed ensembles
#   adaptation  = DAgger in snow: label everything vs. ask only when risky (tau 0.5, 0.9)
#   metadrive   = (optional stage) the same comparison on unseen MetaDrive maps
#   followup    = (stage) fixes after the first full run:
#                 (b) gated ensemble tests re-run with thresholds calibrated on the ensemble itself
#                 (d) control model dinov2_mse_chunk8: MSE head + 8-step chunks, separates "flow" from "chunking"
#                 (c) DAgger redesigned: PilotNet-flow adapting to DESERT (where it fails), scored without the gate
#   followup2   = (stage) reviewer-driven additions:
#                 harder unseen styles fog / night / dust (more failures for the DINOv2 models),
#                 DAgger that queries on the flow head's action spread only (sigma), and
#                 extra failure-prediction baselines (kNN feature distance, MC dropout, autoencoder) on all test laps
#
# Profiles (PROFILE=...):
#   quick   ~30-60 min on a laptop GPU: tiny version of EVERY stage, to check the machine end to end
#   laptop  the paper run, tuned for RTX 4050 (6 GB) + 20-core CPU (~10-14 h; run overnight or stage by stage)
#   cpu     no GPU / no pretrained weights: PilotNet cells only, small
#
# Usage (Linux / WSL / macOS terminal, or Git Bash on Windows, with the venv activated):
#   PROFILE=quick  bash scripts/run_focused.sh
#   PROFILE=laptop bash scripts/run_focused.sh
#   PROFILE=laptop STAGES="closed_loop tables" bash scripts/run_focused.sh     # selected stages only
#   PROFILE=laptop STAGES="metadrive tables"   bash scripts/run_focused.sh     # after: pip install -r requirements-metadrive.txt
#   PROFILE=laptop STAGES="followup tables"    bash scripts/run_focused.sh     # the follow-up fixes (~5-6 h)
#   PROFILE=laptop STAGES="followup2 tables"   bash scripts/run_focused.sh     # harder styles, extra baselines, sigma-only DAgger (~4 h)
# Every step skips work that already exists, so re-running after an interruption continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1          # UTF-8 everywhere (Windows consoles) + live logs

PROFILE=${PROFILE:-laptop}
[ "$PROFILE" = "gpu" ] && PROFILE=laptop
case "$PROFILE" in
  quick)
    SEEDS=${SEEDS:-"0 1"};  TRAIN_TRACKS=${TRAIN_TRACKS:-0-9};  TEST_TRACKS=${TEST_TRACKS:-1000-1002}
    MODELS=${MODELS:-"pilotnet_mse pilotnet_flow dinov2_mse dinov2_flow"};  EPOCHS_MSE=${EPOCHS_MSE:-3};  EPOCHS_FLOW=${EPOCHS_FLOW:-3}
    DAGGER_MODEL=${DAGGER_MODEL:-dinov2_flow};  DAGGER_ROUNDS=${DAGGER_ROUNDS:-1};  DAGGER_TRACKS=${DAGGER_TRACKS:-2}
    DAGGER_EPOCHS=${DAGGER_EPOCHS:-2};  WORKERS=${WORKERS:-8};  JOBS=${JOBS:-5};  CL_ARGS=${CL_ARGS:-"--max_steps 300"}
    MD_TRAIN_MAPS=${MD_TRAIN_MAPS:-0-2};  MD_TEST_MAPS=${MD_TEST_MAPS:-1000-1001};  MD_SEEDS=${MD_SEEDS:-"0"};  MD_STEPS=${MD_STEPS:-300}
    DATA=${DATA:-data_quick};  RUNS=${RUNS:-runs/quick};  OUT=${OUT:-results/quick} ;;
  laptop)
    SEEDS=${SEEDS:-"0 1 2"};  TRAIN_TRACKS=${TRAIN_TRACKS:-0-59};  TEST_TRACKS=${TEST_TRACKS:-1000-1019}
    MODELS=${MODELS:-"pilotnet_mse pilotnet_flow dinov2_mse dinov2_flow"};  EPOCHS_MSE=${EPOCHS_MSE:-25};  EPOCHS_FLOW=${EPOCHS_FLOW:-30}
    DAGGER_MODEL=${DAGGER_MODEL:-dinov2_flow};  DAGGER_ROUNDS=${DAGGER_ROUNDS:-3};  DAGGER_TRACKS=${DAGGER_TRACKS:-10}
    DAGGER_EPOCHS=${DAGGER_EPOCHS:-10};  WORKERS=${WORKERS:-10};  JOBS=${JOBS:-5};  CL_ARGS=${CL_ARGS:-""}
    MD_TRAIN_MAPS=${MD_TRAIN_MAPS:-0-49};  MD_TEST_MAPS=${MD_TEST_MAPS:-1000-1019};  MD_SEEDS=${MD_SEEDS:-"0 1 2"};  MD_STEPS=${MD_STEPS:-1000}
    DATA=${DATA:-data};  RUNS=${RUNS:-runs/focused};  OUT=${OUT:-results/focused} ;;
  cpu)
    SEEDS=${SEEDS:-"0 1 2"};  TRAIN_TRACKS=${TRAIN_TRACKS:-0-19};  TEST_TRACKS=${TEST_TRACKS:-1000-1007}
    MODELS=${MODELS:-"pilotnet_mse pilotnet_flow"};  EPOCHS_MSE=${EPOCHS_MSE:-10};  EPOCHS_FLOW=${EPOCHS_FLOW:-15}
    DAGGER_MODEL=${DAGGER_MODEL:-pilotnet_flow};  DAGGER_ROUNDS=${DAGGER_ROUNDS:-2};  DAGGER_TRACKS=${DAGGER_TRACKS:-3}
    DAGGER_EPOCHS=${DAGGER_EPOCHS:-5};  WORKERS=${WORKERS:-2};  JOBS=${JOBS:-1};  CL_ARGS=${CL_ARGS:-""}
    MD_TRAIN_MAPS=${MD_TRAIN_MAPS:-0-9};  MD_TEST_MAPS=${MD_TEST_MAPS:-1000-1004};  MD_SEEDS=${MD_SEEDS:-"0"};  MD_STEPS=${MD_STEPS:-600}
    DATA=${DATA:-data};  RUNS=${RUNS:-runs/focused};  OUT=${OUT:-results/focused} ;;
  *) echo "unknown PROFILE=$PROFILE (quick | laptop | cpu)"; exit 1 ;;
esac
STAGES=${STAGES:-"data train closed_loop dagger tables"}
CHUNK_MODEL=${CHUNK_MODEL:-dinov2_mse_chunk8}
DAGGER2_MODEL=${DAGGER2_MODEL:-pilotnet_flow};  DAGGER2_STYLE=${DAGGER2_STYLE:-1};  DAGGER2_NAME=${DAGGER2_NAME:-desert}
MD_MODELS=${MD_MODELS:-"pilotnet_mse dinov2_flow"};  MD_JOBS=${MD_JOBS:-2}
SIM=configs/sims/drivelite.yaml
CL=$OUT/closed_loop;  CL_MD=$OUT/closed_loop_metadrive;  LOGS=$OUT/logs
CORES=$(nproc 2>/dev/null || echo 4)
mkdir -p "$RUNS" "$OUT" "$LOGS"
exec > >(tee -a "$LOGS/main.log") 2>&1

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo -e "\n=== [$(date +%H:%M:%S)] $* ==="; }
is_flow() { [[ "$1" == *flow* ]]; }
T0=$(date +%s)
elapsed() { local s=$(( $(date +%s) - T0 )); printf "%dh%02dm" $((s/3600)) $((s%3600/60)); }

# wait for our own background jobs only (a bare `wait` would also wait for the tee logger and hang)
wait_jobs() { local p; for p in $(jobs -p); do wait "$p" 2>/dev/null || true; done; }

# run_job <name> <command...>: run in the background, at most $JOBS at once, output to $LOGS/<name>.log
run_job() {
  local name=$1; shift
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  ( export OMP_NUM_THREADS=$(( CORES / JOBS > 0 ? CORES / JOBS : 1 )) MKL_NUM_THREADS=$(( CORES / JOBS > 0 ? CORES / JOBS : 1 ))
    if "$@" > "$LOGS/$name.log" 2>&1; then echo "  [ok]   $name  ($(elapsed))"; else echo "  [FAIL] $name  -> $LOGS/$name.log"; fi ) &
}

# dagger_done <dir> <rounds>: true if dagger_history.json already contains round <rounds>
dagger_done() { [ -f "$1/dagger_history.json" ] && python - "$1/dagger_history.json" "$2" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
sys.exit(0 if max((r.get("iter", 0) for r in h), default=0) >= int(sys.argv[2]) else 1)
PY
}

log "profile=$PROFILE stages='$STAGES' models='$MODELS' seeds='$SEEDS' cores=$CORES jobs=$JOBS workers=$WORKERS"
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# ------------------------------------------------------------------ 1. data
if has data; then
  log "data: expert demonstrations (DART noise) on style 0 tracks $TRAIN_TRACKS"
  [ -f $DATA/dl_test_shape/driving_log.csv ] || run_job data_test_shape python scripts/collect.py --env drivelite --out $DATA/dl_test_shape --seeds 1000-1004 --style 0 --noise 0
  [ -f $DATA/dl_test_snow/driving_log.csv ]  || run_job data_test_snow  python scripts/collect.py --env drivelite --out $DATA/dl_test_snow  --seeds 1000-1004 --style 2 --noise 0
  [ -f $DATA/dl_train/driving_log.csv ]      || python scripts/collect.py --env drivelite --out $DATA/dl_train --seeds $TRAIN_TRACKS --style 0 --noise 0.15
  wait_jobs
fi

# ------------------------------------------------------------------ 2. train (GPU, one run at a time)
if has train; then
  for m in $MODELS; do for s in $SEEDS; do
    run=$RUNS/${m}_s$s
    [ -f $run/metrics.json ] && continue
    E=$EPOCHS_MSE; is_flow $m && E=$EPOCHS_FLOW
    log "train $m seed $s ($E epochs)  [elapsed $(elapsed)]"
    python scripts/train.py --config configs/focused/$m.yaml $SIM --seed $s --out $run --set name=$m \
      train.epochs=$E train.patience=$E data.num_workers=$WORKERS \
      "data.train_logs=[$DATA/dl_train/driving_log.csv]" \
      "data.test_logs=[$DATA/dl_test_shape/driving_log.csv,$DATA/dl_test_snow/driving_log.csv]"
  done; done
fi

# ------------------------------------------------------------------ 3. closed loop (parallel jobs)
# cl <out-subdir> <model-name> <tag> <style> <gate> <ckpt...>
cl() {
  local sub=$1 name=$2 tag=$3 style=$4 gate=$5; shift 5
  local gargs="--unc_source $gate"; [ "$gate" = "off" ] && gargs="--no_gate"
  local ens=""; [ $# -gt 1 ] && ens="_ens"
  [ -f $CL/$sub/${name}${ens}_${tag}_gate-${gate}/summary.json ] && return 0
  for c in "$@"; do [ -f "$c" ] || { echo "  [skip] $sub $tag: missing $c"; return 0; }; done
  run_job "cl_${sub}_${tag}_${gate}" python scripts/closed_loop_eval.py --env drivelite --style $style --seeds $TEST_TRACKS \
    --out $CL/$sub --tag $tag $gargs $CL_ARGS --ckpt "$@"
}
if has closed_loop; then
  log "closed loop on unseen tracks ($TEST_TRACKS), $JOBS parallel jobs  [elapsed $(elapsed)]"
  for m in $MODELS; do
    if is_flow $m; then GATES_MAIN="off sigma ood max"; else GATES_MAIN="off ood"; fi   # MSE heads have no sigma
    for s in $SEEDS; do
      ck=$RUNS/${m}_s$s/best.pt
      for st in 0:shape 2:snow; do for g in $GATES_MAIN; do cl ${m}_s$s $m ${st#*:} ${st%%:*} $g $ck; done; done
      for st in 1:desert 3:dusk;  do for g in off max;      do cl ${m}_s$s $m ${st#*:} ${st%%:*} $g $ck; done; done
    done
    # deep ensemble of the seeds (epistemic baseline: member disagreement vs. feature novelty)
    set -- $SEEDS
    if [ $# -gt 1 ]; then
      cks=$(for s in $SEEDS; do echo -n "$RUNS/${m}_s$s/best.pt "; done)
      for st in 0:shape 2:snow; do for g in off sigma ood max; do cl ens_$m $m ${st#*:} ${st%%:*} $g $cks; done; done
    fi
  done
  wait_jobs
fi

# ------------------------------------------------------------------ 4. DAgger adaptation to snow
if has dagger; then
  for q in all:0.5 uncertainty:0.5 uncertainty:0.9; do
    mode=${q%%:*}; tau=${q#*:}; d=$RUNS/dagger_${mode}_tau$tau
    dagger_done $d $DAGGER_ROUNDS && continue
    log "DAgger in snow: query=$mode tau=$tau  [elapsed $(elapsed)]"
    python scripts/dagger.py --env drivelite --style 2 --init_ckpt $RUNS/${DAGGER_MODEL}_s0/best.pt \
      --base_logs $DATA/dl_train/driving_log.csv --out $d --iters $DAGGER_ROUNDS --seeds_per_iter $DAGGER_TRACKS \
      --epochs $DAGGER_EPOCHS --query $mode --risk_tau $tau --beta 0.7 --eval_seeds $TEST_TRACKS \
      --train_set data.num_workers=$WORKERS
  done
fi

# ------------------------------------------------------------------ 4b. follow-up fixes
if has followup; then
  # (b) gated ensemble tests: delete the old (mis-calibrated) results once, then re-run them
  if [ ! -f $CL/.ens_recalibrated ]; then
    log "follow-up (b): re-running gated ensemble tests with calibrated thresholds  [elapsed $(elapsed)]"
    rm -rf $CL/ens_*/*_gate-sigma $CL/ens_*/*_gate-ood $CL/ens_*/*_gate-max
    touch $CL/.ens_recalibrated
  fi
  for m in $MODELS; do
    set -- $SEEDS
    if [ $# -gt 1 ]; then
      cks=$(for s in $SEEDS; do echo -n "$RUNS/${m}_s$s/best.pt "; done)
      for st in 0:shape 2:snow; do for g in sigma ood max; do cl ens_$m $m ${st#*:} ${st%%:*} $g $cks; done; done
    fi
  done
  # (d) chunking control (trains on the GPU while the ensemble tests run)
  for s in $SEEDS; do
    run=$RUNS/${CHUNK_MODEL}_s$s
    [ -f $run/metrics.json ] && continue
    log "follow-up (d): train $CHUNK_MODEL seed $s ($EPOCHS_MSE epochs)  [elapsed $(elapsed)]"
    python scripts/train.py --config configs/focused/$CHUNK_MODEL.yaml $SIM --seed $s --out $run --set name=$CHUNK_MODEL \
      train.epochs=$EPOCHS_MSE train.patience=$EPOCHS_MSE data.num_workers=$WORKERS \
      "data.train_logs=[$DATA/dl_train/driving_log.csv]" \
      "data.test_logs=[$DATA/dl_test_shape/driving_log.csv,$DATA/dl_test_snow/driving_log.csv]"
  done
  for s in $SEEDS; do
    for st in 0:shape 1:desert 2:snow 3:dusk; do cl ${CHUNK_MODEL}_s$s $CHUNK_MODEL ${st#*:} ${st%%:*} off $RUNS/${CHUNK_MODEL}_s$s/best.pt; done
  done
  wait_jobs
  # (c) DAgger redesigned: a model that really fails in the new style, scored without the speed gate
  for q in all:0.5 uncertainty:0.5 uncertainty:0.9; do
    mode=${q%%:*}; tau=${q#*:}; d=$RUNS/dagger2_${DAGGER2_NAME}_${mode}_tau$tau
    dagger_done $d $DAGGER_ROUNDS && continue
    rm -rf $d
    log "follow-up (c): DAgger in $DAGGER2_NAME from $DAGGER2_MODEL: query=$mode tau=$tau  [elapsed $(elapsed)]"
    python scripts/dagger.py --env drivelite --style $DAGGER2_STYLE --init_ckpt $RUNS/${DAGGER2_MODEL}_s0/best.pt \
      --base_logs $DATA/dl_train/driving_log.csv --out $d --iters $DAGGER_ROUNDS --seeds_per_iter $DAGGER_TRACKS \
      --epochs $DAGGER_EPOCHS --query $mode --risk_tau $tau --beta 0.7 --eval_seeds $TEST_TRACKS \
      --train_set data.num_workers=$WORKERS
  done
fi

# ------------------------------------------------------------------ 4c. follow-up 2
if has followup2; then
  log "follow-up 2 (a): harder styles fog / night / dust, all models, no gate  [elapsed $(elapsed)]"
  for m in $MODELS $CHUNK_MODEL; do
    for s in $SEEDS; do
      for st in 4:fog 5:night 6:dust; do cl ${m}_s$s $m ${st#*:} ${st%%:*} off $RUNS/${m}_s$s/best.pt; done
    done
  done
  wait_jobs
  for tau in 0.5 0.9; do
    d=$RUNS/dagger2_${DAGGER2_NAME}_uncsigma_tau$tau
    dagger_done $d $DAGGER_ROUNDS && continue
    rm -rf $d
    log "follow-up 2 (b): DAgger in $DAGGER2_NAME, query on action spread only, tau=$tau  [elapsed $(elapsed)]"
    python scripts/dagger.py --env drivelite --style $DAGGER2_STYLE --init_ckpt $RUNS/${DAGGER2_MODEL}_s0/best.pt \
      --base_logs $DATA/dl_train/driving_log.csv --out $d --iters $DAGGER_ROUNDS --seeds_per_iter $DAGGER_TRACKS \
      --epochs $DAGGER_EPOCHS --query uncertainty --unc_source sigma --risk_tau $tau --beta 0.7 --eval_seeds $TEST_TRACKS \
      --train_set data.num_workers=$WORKERS
  done
  log "follow-up 2 (c): extra failure-prediction baselines on all logged test laps  [elapsed $(elapsed)]"
  python scripts/extra_signals.py --runs $RUNS --cl $CL --data $DATA --out $OUT/extra_signals \
    --models $MODELS $CHUNK_MODEL --seeds $SEEDS --workers $WORKERS
fi

# ------------------------------------------------------------------ 5. MetaDrive check (optional stage)
if has metadrive; then
  python -c "import metadrive" 2>/dev/null || { echo "MetaDrive is not installed: pip install -r requirements-metadrive.txt (Python <= 3.11)"; exit 1; }
  log "MetaDrive: collect maps $MD_TRAIN_MAPS, test on unseen maps $MD_TEST_MAPS  [elapsed $(elapsed)]"
  JOBS_SAVE=$JOBS; JOBS=$MD_JOBS
  [ -f $DATA/md_test_maps/driving_log.csv ] || run_job md_test_maps python scripts/collect.py --env metadrive --out $DATA/md_test_maps --seeds 1000-1004 --noise 0 --max_steps $MD_STEPS
  [ -f $DATA/md_train/driving_log.csv ]     || run_job md_train python scripts/collect.py --env metadrive --out $DATA/md_train --seeds $MD_TRAIN_MAPS --noise 0.15 --max_steps $MD_STEPS
  wait_jobs
  [ -f $DATA/md_train/driving_log.csv ] || { echo "MetaDrive data collection failed - see $LOGS/md_train.log"; exit 1; }
  for m in $MD_MODELS; do for s in $MD_SEEDS; do
    run=$RUNS/md_${m}_s$s
    [ -f $run/metrics.json ] && continue
    E=$EPOCHS_MSE; is_flow $m && E=$EPOCHS_FLOW
    log "train (MetaDrive) $m seed $s"
    python scripts/train.py --config configs/focused/$m.yaml configs/sims/metadrive.yaml --seed $s --out $run \
      --set name=md_$m train.epochs=$E train.patience=$E data.num_workers=$WORKERS \
      "data.train_logs=[$DATA/md_train/driving_log.csv]" "data.test_logs=[$DATA/md_test_maps/driving_log.csv]"
  done; done
  for m in $MD_MODELS; do for s in $MD_SEEDS; do for g in off max; do
    gargs="--unc_source $g"; [ "$g" = "off" ] && gargs="--no_gate"
    [ -f $CL_MD/md_${m}_s$s/md_${m}_maps_gate-$g/summary.json ] && continue
    run_job md_cl_${m}_s${s}_$g python scripts/closed_loop_eval.py --env metadrive --seeds $MD_TEST_MAPS --max_steps $MD_STEPS \
      --out $CL_MD/md_${m}_s$s --tag maps $gargs --no_expert --ckpt $RUNS/md_${m}_s$s/best.pt
  done; done; done
  wait_jobs
  JOBS=$JOBS_SAVE
fi

# ------------------------------------------------------------------ 6. tables + figures
if has tables; then
  log "tables"
  python scripts/aggregate.py --runs $RUNS --split test --dataset dl_test_shape --out $OUT/offline_shape.tex | tee $OUT/offline_shape.md || true
  python scripts/aggregate.py --runs $RUNS --split test --dataset dl_test_snow  --out $OUT/offline_snow.tex  | tee $OUT/offline_snow.md || true
  python scripts/aggregate_closed_loop.py --dir "$CL/*" --out $OUT/closed_loop.tex | tee $OUT/closed_loop.md || true
  ls -d $RUNS/dagger* >/dev/null 2>&1 && python scripts/summarize_dagger.py $RUNS/dagger* --out $OUT/dagger.md || true
  ls $CL_MD/* >/dev/null 2>&1 && python scripts/aggregate_closed_loop.py --dir "$CL_MD/*" --out $OUT/closed_loop_metadrive.tex | tee $OUT/closed_loop_metadrive.md || true
  python scripts/focused_report.py --runs $RUNS --cl "$CL" --out $OUT
fi
log "done in $(elapsed) -> $OUT  (start with $OUT/REPORT.md)"
