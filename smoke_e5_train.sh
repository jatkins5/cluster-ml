#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J e5smoke
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/smoke_e5_train_%j.out
# E5/E4 one-epoch smoke (review rounds 16-17) -- diagnostics only.
# PREPARED; run only with the user's authorisation.
#  1. legacy call (no --loss/--label-curves) still runs; saves loss=huber
#  2. h/m/d/c x seeds 42,43 at 1 epoch (c: 1 pretraining epoch) through the
#     real trainer; full per-run logs and wall times kept
#  3. compare_e5.py --mode smoke on that fixture (diagnostic output only)
# Outputs go to a job-specific directory that must not already exist.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_e5_${SLURM_JOB_ID:?run under SLURM}
if [ -e "$T" ]; then echo "refusing: $T exists"; exit 1; fi
mkdir "$T"
COMMON="--dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz --train-view pooled
        --select-view single --epochs 1 --cl-epochs 1"   # cl-epochs inert unless --contrastive-pretrain
echo "======== 1. legacy call"
$PY train_cnn_mock.py $COMMON --seed 42 --out-prefix $T/legacy > $T/legacy.log 2>&1
$PY -c "import numpy as np; d=np.load('$T/legacy_preds.npz'); assert str(d['loss'])=='huber' and d['class_of'].size==0 and not bool(d['contrastive_pretrain']); print('  legacy: loss huber, no label curves, no pretraining')"
echo "======== 2. four arms x two seeds, 1 epoch"
for S in 42 43; do
  for A in e5_h:"--loss huber" e5_m:"--loss mse" e5_d:"--loss class7" \
           e4_c:"--loss huber --contrastive-pretrain"; do
    t0=$(date +%s)
    $PY train_cnn_mock.py $COMMON --seed $S ${A#*:} \
        --label-curves dataset_nh4_144_s9991_128.h5 --out-prefix $T/${A%%:*}_s$S \
        > $T/${A%%:*}_s$S.log 2>&1
    echo "  ${A%%:*} seed $S: $(( $(date +%s) - t0 )) s  ($(grep -c 'contrastive pretraining' $T/${A%%:*}_s$S.log) pretraining folds)"
  done
done
grep -h "contrastive pretraining" $T/e4_c_s42.log || true
echo "======== 3. scorer (smoke mode)"
$PY compare_e5.py --mode smoke --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d \
    --pred-dir $T --seeds 42,43 --expect-epochs 1 --expect-cl-epochs 1 --boot 200
echo "E5/E4 TRAINING SMOKE PASSED ($T kept)"
