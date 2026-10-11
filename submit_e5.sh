#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 06:00:00
#SBATCH -J e5e4
#SBATCH --array=0-19%2
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/e5e4_%A_%a.out
# E5 + E4 (review rounds 15-17), ONE reviewed source revision for all arms
# (compare_e5.py requires identical source hashes). E0 X-ray-only anchor:
# same dataset, manifest, pooled training, single-view inner selection,
# 60 epochs, seeds 42-46; label curves passed to every arm.
#    0-4  h  huber (matched frozen-source control)
#    5-9  m  mse
#   10-14 d  class7
#   15-19 c  huber + contrastive pretraining (300 epochs, fitting halos)
# PREPARED, NOT SUBMITTED.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
case $((I / 5)) in
  0) O=e5_h; X="--loss huber" ;;
  1) O=e5_m; X="--loss mse" ;;
  2) O=e5_d; X="--loss class7" ;;
  3) O=e4_c; X="--loss huber --contrastive-pretrain" ;;
esac
if [ -e cnn_preds/${O}_s${S}_preds.npz ]; then echo "refusing to overwrite ${O}_s$S"; exit 1; fi
./venv/bin/python -u train_cnn_mock.py --dataset xray_fix_s9991.h5 \
    --fold-manifest folds_e0.npz --train-view pooled --select-view single \
    --seed $S $X --label-curves dataset_nh4_144_s9991_128.h5 \
    --out-prefix cnn_preds/${O}_s$S
