#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J b2grid2
#SBATCH -a 0-11%2
#SBATCH -o logs/b2_grid2_%A_%a.out
# Seeds 44-46 for the same four conditions. The two-seed run put the
# headline increment (image+mass over mass-only) at +0.091 with per-seed
# values of +0.120 and +0.063, which is too wide a spread to quote.
cd /oscar/data/idellant/cluster-ml
I=$SLURM_ARRAY_TASK_ID
SEED=$((44 + I % 3))
case $((I / 3)) in
  0) NAME=img;      FLAGS="" ;;
  1) NAME=imgmass;  FLAGS="--mass" ;;
  2) NAME=imgmassn; FLAGS="--mass --mass-noise" ;;
  3) NAME=massonly; FLAGS="--mass --no-image" ;;
esac
echo "condition=$NAME seed=$SEED flags='$FLAGS'"
./venv/bin/python -u train_cnn_mock.py \
    --dataset injected_nh4.h5 --seed "$SEED" $FLAGS \
    --out-prefix "cnn_preds/b2_${NAME}_s${SEED}"
