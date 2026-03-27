#!/bin/bash
#SBATCH --job-name=cnn-merger-tsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_merger_tsc_%j.out
#SBATCH --error=logs/cnn_merger_tsc_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

python train_cnn_pooled.py \
    --folds 5 \
    --epochs 60 \
    --batch-size 32 \
    --merger-tsc
