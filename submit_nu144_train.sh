#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 03:00:00
#SBATCH -J nu144tr
#SBATCH --array=0-10
#SBATCH -o logs/nu144_train_%A_%a.out
# 144 MHz emission vs the 1.4 GHz nh4 baseline, same seeds as the runs it is
# paired with: tasks 0-4 the clean pooled CNN (submit_a3_baselines.sh, inner
# selection, seeds 42-46); 5-7 the transfer model image-only and 8-10 image +
# noisy mass (submit_b23_train.sh, seeds 42-44).
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 5 ]; then
  SEED=$((42 + I))
  ./venv/bin/python -u train_cnn_pooled.py \
      --dataset dataset_nh4_144_128.h5 --pseudo-tsc --seed "$SEED" \
      --select inner --save-preds "cnn_preds/pooled_nh4_144_inner_s${SEED}.npz"
else
  J=$((I - 5))
  SEED=$((42 + J % 3))
  if [ $J -lt 3 ]; then NAME=sfimg; FLAGS=""; else NAME=sfimgmassn; FLAGS="--mass --mass-noise"; fi
  ./venv/bin/python -u train_cnn_mock.py \
      --dataset injected_nh4_144_simflux.h5 --seed "$SEED" $FLAGS \
      --out-prefix "cnn_preds/nu144_${NAME}_s${SEED}"
fi
