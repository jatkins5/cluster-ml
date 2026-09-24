#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 01:00:00
#SBATCH -J xraycnn
#SBATCH -a 0-9%2
#SBATCH -o logs/xray_cnn_%A_%a.out
# X-ray-only pooled CNN, current protocol (inner-split selection), same 5
# seeds as the radio headline: original stretch vs the scaled fix.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 5))
if [ $I -lt 5 ]; then NAME=orig; else NAME=scaled; fi
./venv/bin/python -u train_cnn_pooled.py \
    --dataset "dataset_xray_128_${NAME}.h5" --pseudo-tsc --seed "$SEED" \
    --select inner --save-preds "cnn_preds/xray_${NAME}_inner_s${SEED}.npz"
