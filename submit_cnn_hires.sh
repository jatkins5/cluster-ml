#!/bin/bash
#SBATCH --job-name=cnn-hires
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_hires_%j.out
#SBATCH --error=logs/cnn_hires_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== 256px, 4×R500c (~30-40 kpc/pixel) ==="
python train_cnn.py \
    --folds 5 \
    --epochs 120 \
    --batch-size 16 \
    --merger-tsc \
    --huber-delta 2.0 \
    --dataset dataset_256.h5

echo ""
echo "=== 256px, 2×R500c (~15-20 kpc/pixel) ==="
python train_cnn.py \
    --folds 5 \
    --epochs 120 \
    --batch-size 16 \
    --merger-tsc \
    --huber-delta 2.0 \
    --dataset dataset_256_2r500.h5
