#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 01:30:00
#SBATCH -J xrtrain
#SBATCH -a 0-5%2
#SBATCH -o logs/xray_real_train_%A_%a.out
# X-ray-only pooled CNN at realistic Chandra depths, same protocol as the
# radio transfer runs (grouped folds, inner-split selection, 3 realizations
# scored as the cluster mean).
cd /oscar/data/idellant/cluster-ml
DEPTHS=(1 3)
I=$SLURM_ARRAY_TASK_ID
D=${DEPTHS[$((I / 3))]}
SEED=$((42 + I % 3))
./venv/bin/python -u train_cnn_mock.py --dataset "xray_real_${D}.h5" \
    --seed "$SEED" --out-prefix "cnn_preds/xreal_${D}_s${SEED}"
