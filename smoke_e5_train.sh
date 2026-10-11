#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J e5smoke
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/smoke_e5_train_%j.out
# E5 one-epoch smoke (review round 16) -- diagnostics only, not performance.
# PREPARED, NOT SUBMITTED: needs the user's authorisation.
#  1. legacy call (no --loss, no --label-curves) still runs; loss saved huber
#  2. h/m/d x seeds 42,43 at 1 epoch through the real trainer
#  3. compare_e5.py accepts the fixture (--expect-epochs 1) and emits the
#     probabilistic block; wall time per run recorded for the E4 estimate
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_e5_tmp; rm -rf $T; mkdir -p $T
COMMON="--dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz --train-view pooled
        --select-view single --epochs 1"
echo "======== 1. legacy call"
$PY train_cnn_mock.py $COMMON --seed 42 --out-prefix $T/legacy | grep -E "OOF R2 \(ONE"
$PY -c "import numpy as np; d=np.load('$T/legacy_preds.npz'); assert str(d['loss'])=='huber' and d['class_of'].size==0; print('  legacy: loss huber, no label curves')"
echo "======== 2. three arms x two seeds, 1 epoch"
for S in 42 43; do
  for A in h:huber m:mse d:class7; do
    t0=$(date +%s)
    $PY train_cnn_mock.py $COMMON --seed $S --loss ${A#*:} \
        --label-curves dataset_nh4_144_s9991_128.h5 --out-prefix $T/e5_${A%%:*}_s$S \
        | grep -E "label curves|OOF R2 \(ONE"
    echo "  ${A%%:*} seed $S: $(( $(date +%s) - t0 )) s"
  done
done
echo "======== 3. scorer on the fixture"
$PY compare_e5.py --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d \
    --pred-dir $T --seeds 42,43 --expect-epochs 1 --boot 200
echo "E5 TRAINING SMOKE PASSED"
