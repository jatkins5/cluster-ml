#!/bin/bash
#SBATCH --job-name=camels-int
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/camels_internal_%j.out
#SBATCH --error=logs/camels_internal_%j.err

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u train_cnn_camels.py --mode transfer --tag internal \
    --camels dataset_camels_128.h5 --camels-tsc-max 8.0 --epochs 80
