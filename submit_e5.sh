#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J e5
#SBATCH --array=0-14%2
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/e5_%A_%a.out
# E5 (review rounds 15-16): loss/head arms on the E0 X-ray-only anchor.
# Same dataset, manifest, pooled training, single-view inner selection,
# 60 epochs, seeds 42-46; only --loss differs. The label curves are passed
# to every arm (identity metadata). Scored by compare_e5.py.
#    0-4 huber (h)   5-9 mse (m)   10-14 class7 (d)
# PREPARED, NOT SUBMITTED. Source freeze: run together with E4 from one
# reviewed source revision (compare_e5 requires identical source hashes).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
case $((I / 5)) in 0) A=h; L=huber ;; 1) A=m; L=mse ;; 2) A=d; L=class7 ;; esac
./venv/bin/python -u train_cnn_mock.py --dataset xray_fix_s9991.h5 \
    --fold-manifest folds_e0.npz --train-view pooled --select-view single \
    --seed $S --loss $L --label-curves dataset_nh4_144_s9991_128.h5 \
    --out-prefix cnn_preds/e5_${A}_s$S
