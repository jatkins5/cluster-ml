#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J lblmse
#SBATCH --array=0-4%5
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/lbl_mse_%A_%a.out
# Phase 4 nominated follow-up (review rounds 33-36): ONE X-ray arm trained on
# the capped catalogue TSC (T_coll) with MSE instead of Huber. Everything else
# as the accepted lbl_x_tsc_tcoll Huber arm: dataset, manifest/inner sets,
# capped labels and no-event convention, architecture, 60 epochs, optimiser,
# pooled training, single-view inner selection, seeds 42-46. Scored against
# the existing Huber artifacts by compare_lbl_mse.py.
# Estimated 5 x ~9-11 min on one GPU each (from array 8427455 tasks 5-9).
# PREPARED; submit only with the user's authorisation.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
S=$((42 + SLURM_ARRAY_TASK_ID))
O=cnn_preds/lbl_x_tsc_tcoll_mse_s${S}
if [ -e ${O}_preds.npz ]; then echo "refusing to overwrite $O"; exit 1; fi
./venv/bin/python -u train_cnn_mock.py --dataset xray_fix_s9991.h5 --fold-manifest folds_e0.npz \
    --train-view pooled --select-view single --seed $S --label tsc_tcoll --loss mse \
    --tsc-dir /oscar/data/idellant/Chuiyang/TSC_new --out-prefix $O
