#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J labels
#SBATCH --array=0-19%6
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/labels_%A_%a.out
# Training-label comparison (2026-11-05, review rounds 22-23): pseudo-TSC vs
# Chuiyang's catalogue TSC from T_coll (TSC_new: tsc_gyr_from_Tcoll), the
# capped time since the last qualifying catalogue collision (NaN -> 4 Gyr).
# Each arm selects checkpoints on its own target (inner single-view R2).
# Same data, folds (folds_e0.npz), protocol and seeds 42-46 within each
# modality; only --label differs. Every arm passes --tsc-dir so all models
# carry the same truths. Scored by compare_labels.py. PREPARED, NOT SUBMITTED.
#    0-9   X-ray (xray_fix_s9991.h5):   pseudo 0-4, tsc_tcoll 5-9
#   10-19  radio 3 Mpc, new model (injected_syn_phys_b3000_s9991.h5): same order
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
LABS=(pseudo tsc_tcoll); LAB=${LABS[$(((I % 10) / 5))]}
if [ $I -lt 10 ]; then M=x; D=xray_fix_s9991.h5; else M=r; D=injected_syn_phys_b3000_s9991.h5; fi
O=cnn_preds/lbl_${M}_${LAB}_s${S}
if [ -e ${O}_preds.npz ]; then echo "refusing to overwrite $O"; exit 1; fi
./venv/bin/python -u train_cnn_mock.py --dataset $D --fold-manifest folds_e0.npz \
    --train-view pooled --select-view single --seed $S --label $LAB \
    --tsc-dir /oscar/data/idellant/Chuiyang/TSC_new --out-prefix $O
