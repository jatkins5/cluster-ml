#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=64G -t 02:00:00
#SBATCH -J lxtrain
#SBATCH -a 0-5%2
#SBATCH -o logs/lx_train_%A_%a.out
#   0-2 X-ray only (predicts all 15 real Chandra clusters)
#   3-5 joint radio + X-ray (predicts the clusters with both)
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 3))
if [ $I -lt 3 ]; then
  ./venv/bin/python -u train_cnn_mock.py --dataset xray_real_placed_lx.h5 \
      --seed "$SEED" --out-prefix "cnn_preds/xlx_s${SEED}"
else
  ./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_simflux.h5 \
      --xray-dataset xray_real_placed_lx.h5 --seed "$SEED" \
      --out-prefix "cnn_preds/jointlx_s${SEED}"
fi
