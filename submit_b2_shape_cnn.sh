#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 01:00:00
#SBATCH -J shapecnn
#SBATCH -a 0-4%2
#SBATCH -o logs/shape_cnn_%A_%a.out
# B2 option 4: the same pooled CNN on maps rescaled to a common total flux.
# Seeds match the full-flux run (0.487 +- 0.033) so the comparison is paired.
cd /oscar/data/idellant/cluster-ml
SEED=$((42 + SLURM_ARRAY_TASK_ID))
./venv/bin/python -u train_cnn_pooled.py \
    --dataset dataset_nh4_128_shape.h5 --pseudo-tsc --seed "$SEED" \
    --select inner --save-preds "cnn_preds/pooled_shape_inner_s${SEED}.npz"
