#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J lblsmoke
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/smoke_labels_train_%j.out
# One-epoch smoke for the label comparison (diagnostic only): both modalities,
# two labels (pseudo, tsc_tcoll), seeds 42,43, then compare_labels.py --mode smoke. Also checks
# that a call without --label/--tsc-dir still trains on pseudo-TSC.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
PY="./venv/bin/python -u"
T=smoke_labels_${SLURM_JOB_ID:?run under SLURM}
if [ -e "$T" ]; then echo "refusing: $T exists"; exit 1; fi
mkdir "$T"
COMMON="--fold-manifest folds_e0.npz --train-view pooled --select-view single --epochs 1"
$PY train_cnn_mock.py --dataset xray_fix_s9991.h5 $COMMON --seed 42 --out-prefix $T/legacy > $T/legacy.log 2>&1
$PY -c "import numpy as np; d=np.load('$T/legacy_preds.npz'); assert str(d['label_name'])=='pseudo' and d['truth_tcoll_capped'].size==0; print('  legacy: pseudo-TSC, no catalogue truths')"
for S in 42 43; do
  for M in x:xray_fix_s9991.h5 r:injected_syn_phys_b3000_s9991.h5; do
    for L in pseudo tsc_tcoll; do
      $PY train_cnn_mock.py --dataset ${M#*:} $COMMON --seed $S --label $L \
          --tsc-dir /oscar/data/idellant/Chuiyang/TSC_new \
          --out-prefix $T/lbl_${M%%:*}_${L}_s$S > $T/lbl_${M%%:*}_${L}_s$S.log 2>&1
      grep -h "catalogue TSC joined" $T/lbl_${M%%:*}_${L}_s$S.log | head -1
    done
  done
done
$PY compare_labels.py --mode smoke --expect-epochs 1 --seeds 42,43 --boot 200 --pred-dir $T \
    --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d
echo "LABEL SMOKE PASSED ($T kept)"
