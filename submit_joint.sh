#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 01:00:00
#SBATCH -J joint
#SBATCH -a 0-9%2
#SBATCH -o logs/joint_%A_%a.out
# Joint radio + X-ray pooled CNN, same 5 seeds, folds and inner-split
# selection as the radio-only and X-ray-only runs it is compared against.
#   0-4 fixed X-ray stretch   5-9 original stretch (the old "no help" setup)
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 5))
if [ $I -lt 5 ]; then NAME=scaled; else NAME=orig; fi
./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_128.h5 \
    --xray-dataset "dataset_xray_128_${NAME}.h5" --pseudo-tsc --seed "$SEED" \
    --select inner --save-preds "cnn_preds/joint_${NAME}_inner_s${SEED}.npz"
