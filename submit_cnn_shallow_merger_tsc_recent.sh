#!/bin/bash
#SBATCH --job-name=cnn-mtsc-recent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_shallow_merger_tsc_recent_%j.out
#SBATCH --error=logs/cnn_shallow_merger_tsc_recent_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

python train_cnn.py \
    --folds 5 \
    --epochs 120 \
    --batch-size 32 \
    --merger-tsc \
    --huber-delta 0.5 \
    --tsc-max 2.0
