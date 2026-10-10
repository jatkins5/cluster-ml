#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J boxjoint
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/box_joint_%A_%a.out
# Joint radio + X-ray on 2 Mpc (0-4) and 3 Mpc (5-9) radio boxes, X-ray at
# the recalibrated brightness, snapshots 99+91, seeds 42-46; paired with
# lxcal_joint (1 Mpc, same X-ray calibration).
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 5 ]; then B=2000; else B=3000; fi
S=$((42 + I % 5))
./venv/bin/python -u train_cnn_mock.py \
    --dataset injected_nh4_144_s9991_b${B}_simflux.h5 \
    --xray-dataset xray_placed_144_s9991_b${B}_lxcal.h5 \
    --seed $S --out-prefix cnn_preds/b${B}_joint_lxcal_s$S
