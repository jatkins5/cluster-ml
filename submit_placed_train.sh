#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=64G -t 02:00:00
#SBATCH -J plctrain
#SBATCH -a 0-5%2
#SBATCH -o logs/placed_train_%A_%a.out
# Redshift-placed X-ray (real target z and ACIS-I exposure, common aperture):
#   0-2 X-ray only    3-5 joint with the injected radio mocks
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((42 + I % 3))
if [ $I -lt 3 ]; then
  ./venv/bin/python -u train_cnn_mock.py --dataset xray_real_placed.h5 \
      --seed "$SEED" --out-prefix "cnn_preds/xplaced_s${SEED}"
else
  ./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_simflux.h5 \
      --xray-dataset xray_real_placed.h5 --seed "$SEED" \
      --out-prefix "cnn_preds/jointplaced_s${SEED}"
fi
