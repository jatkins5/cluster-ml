#!/bin/bash
#SBATCH --job-name=cnn-camels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_camels_%j.out
#SBATCH --error=logs/cnn_camels_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python

echo "### TNG only (baseline) ###"
$PY -u train_cnn_camels.py --tag tng_only --epochs 80

echo
echo "### TNG + CAMELS (TSC <= 8 Gyr, M200 > 1e14) ###"
$PY -u train_cnn_camels.py --tag tng_camels --epochs 80 \
    --camels dataset_camels_128.h5 --camels-tsc-max 8.0
