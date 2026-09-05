#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J cnn_weighted
#SBATCH -o logs/cnn_weighted_%j.out

cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
ARGS="--merger-tsc --dataset dataset.h5 --epochs 80"

# Unweighted control run under the identical seed/epochs as the weighted run,
# so the comparison isn't against a baseline trained with different settings.
echo "### unweighted control ###"
$PY -u train_cnn.py $ARGS

echo
echo "### shape-only L_X importance weights ###"
$PY -u train_cnn.py $ARGS --sample-weights sim_obs_shape_weights.npz
