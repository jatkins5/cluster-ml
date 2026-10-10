#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 02:00:00
#SBATCH -J xdepthf22
#SBATCH --array=0-17%2
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/xdepth_fix2_%A_%a.out
# X-ray-only models for the depth figure, corrected count model, snapshot 99,
# seeds 42-44: 0-14 fixed depths 5/20/50/150/500 ks; 15-17 each mock at its
# target's real exposure (the reference band).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
DEPTHS=(5 20 50 150 500)
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 3))
if [ $I -lt 15 ]; then
  KS=${DEPTHS[$((I / 3))]}
  ./venv/bin/python -u train_cnn_mock.py --dataset xray_depthfix2_${KS}ks.h5 \
      --seed $S --out-prefix cnn_preds/xdepthfix2_${KS}ks_s$S
else
  ./venv/bin/python -u train_cnn_mock.py --dataset xray_depthfix2_real.h5 \
      --seed $S --out-prefix cnn_preds/xdepthfix2_real_s$S
fi
