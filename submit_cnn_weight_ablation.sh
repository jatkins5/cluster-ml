#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J cnn_wabl
#SBATCH -o logs/cnn_wabl_%j.out

cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
ARGS="--merger-tsc --dataset dataset.h5 --epochs 80"
W="sim_obs_shape_weights.npz"

# Unweighted training, scored on the obs-like weighted metric. This is the
# number the weighted run has to beat; without it, the weighted run's 0.516
# weighted score cannot be told apart from "the weighted metric is just easier".
echo "### unweighted training, weighted evaluation ###"
$PY -u train_cnn.py $ARGS --sample-weights $W --weights-eval-only
