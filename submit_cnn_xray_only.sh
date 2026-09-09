#!/bin/bash
#SBATCH --job-name=cnn-xray
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_xray_%j.out
#SBATCH --error=logs/cnn_xray_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== X-ray-only shallow CNN, merger-TSC, 128px ==="
python train_cnn.py \
    --dataset dataset_xray_128.h5 \
    --merger-tsc \
    --huber-delta 2.0 \
    --epochs 120 \
    --batch-size 32
