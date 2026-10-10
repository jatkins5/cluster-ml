#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 03:00:00
#SBATCH -J f7train
#SBATCH --array=0-9
#SBATCH --exclude=gpu3001,gpu2504
#SBATCH -o logs/f7_train_%A_%a.out
# Review F7, under the current protocol (inner-split selection), 144 MHz,
# seeds 42-46, paired with pooled_nh4_144_inner_s42-46:
#   0-4  no density cut (dataset_nocut_144_128)  -> is the cut still worth it?
#   5-9  clean joint radio + X-ray on the 144 MHz radio maps (was 1.4 GHz)
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
S=$((42 + I % 5))
if [ $I -lt 5 ]; then
  ./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nocut_144_128.h5 \
      --pseudo-tsc --seed $S --select inner --save-preds cnn_preds/pooled_nocut144_inner_s$S.npz
else
  ./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_144_128.h5 \
      --xray-dataset dataset_xray_128_scaled.h5 --pseudo-tsc --seed $S \
      --select inner --save-preds cnn_preds/joint144_clean_inner_s$S.npz
fi
