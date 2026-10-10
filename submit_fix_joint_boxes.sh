#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 05:00:00
#SBATCH -J fixjbox
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/fix_joint_boxes_%A_%a.out
# Joint radio + X-ray for the 2 Mpc (0-4) and 3 Mpc (5-9) physical radio
# sets, each with X-ray mocks placed on its own redshift draws.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 5 ]; then B=2000; else B=3000; fi
S=$((42 + I % 5))
./venv/bin/python -u train_cnn_mock.py --dataset injected_fix_phys_b${B}_s9991.h5 \
    --xray-dataset xray_fix_b${B}_s9991.h5 --seed $S \
    --out-prefix cnn_preds/fix_phys_b${B}_joint_s$S
