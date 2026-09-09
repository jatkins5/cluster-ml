#!/bin/bash
#SBATCH --job-name=cnn-xray256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/cnn_xray_256_%j.out
#SBATCH --error=logs/cnn_xray_256_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== X-ray-only CNN, 256px, merger-TSC ==="
python train_cnn.py \
    --dataset dataset_xray_256.h5 \
    --merger-tsc \
    --huber-delta 2.0 \
    --epochs 120 \
    --batch-size 16

echo ""
echo "=== Dual-encoder CNN, radio 128px + X-ray 256px ==="
python train_cnn_dual.py \
    --radio-dataset dataset.h5 \
    --xray-dataset dataset_xray_256.h5 \
    --merger-tsc \
    --huber-delta 2.0 \
    --epochs 120 \
    --batch-size 16
