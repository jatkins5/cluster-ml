#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J boxtrain
#SBATCH --array=0-19
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/box_train_%A_%a.out
# Fix 1: realistic radio (0-4, 10-14) and radio + noisy mass (5-9, 15-19)
# on 2 Mpc (0-9) and 3 Mpc (10-19) boxes, snapshots 99+91, seeds 42-46;
# paired with s9991_sfimg / s9991_sfimgmassn (1 Mpc).
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 10 ]; then B=2000; else B=3000; fi
J=$((I % 10)); S=$((42 + J % 5))
D=injected_nh4_144_s9991_b${B}_simflux.h5
if [ $J -lt 5 ]; then
  ./venv/bin/python -u train_cnn_mock.py --dataset $D --seed $S \
      --out-prefix cnn_preds/b${B}_sfimg_s$S
else
  ./venv/bin/python -u train_cnn_mock.py --dataset $D --seed $S --mass --mass-noise \
      --out-prefix cnn_preds/b${B}_sfimgmassn_s$S
fi
