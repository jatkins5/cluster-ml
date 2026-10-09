#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J lxcaltr
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/lxcal_train_%A_%a.out
# X-ray-only (0-4) and joint (5-9) on snapshots 99+91 with the recalibrated
# X-ray brightness; paired with s9991_x / s9991_joint seeds 42-46.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
XB=xray_placed_144_s9991_lxcal.h5
if [ $I -lt 5 ]; then
  ./venv/bin/python -u train_cnn_mock.py --dataset $XB --seed $S \
      --out-prefix cnn_preds/lxcal_x_s$S
else
  ./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_144_s9991_simflux.h5 \
      --xray-dataset $XB --seed $S --out-prefix cnn_preds/lxcal_joint_s$S
fi
