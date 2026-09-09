#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J a3trans
#SBATCH -a 0-3%2
#SBATCH -o logs/a3_trans_%A_%a.out
# A3: re-run the field-injection transfer model with the fixed optimizer
# (AdamW 3e-4 + cosine, was Adam 1e-3 flat) and honest checkpoint selection.
# Replaces mock OOF R2 = 0.426, which carried a measured +0.32 selection bias.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 2))
if [ $I -lt 2 ]; then SEL=inner; else SEL=final; fi
echo "seed=$SEED select=$SEL"
./venv/bin/python -u train_cnn_mock.py \
    --dataset injected_nh4.h5 --seed "$SEED" --select "$SEL" \
    --out-prefix "cnn_preds/injtrans_v2_${SEL}_s${SEED}"
