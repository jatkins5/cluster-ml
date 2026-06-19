#!/bin/bash
#SBATCH --job-name=cnn-aug-oof
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_aug_oof_%j.out
#SBATCH --error=logs/cnn_aug_oof_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

DATA=diffusion_radio_128_v2.h5
AUG=diffusion_out_cond_128_ada/samples_cond.npz

echo "=== Baseline: 5-fold OOF, no synthetic data, select-by overall ==="
python train_cnn_aug_oof.py --data $DATA --ch 48 --epochs 80 \
    --select-by overall --tag baseline

echo ""
echo "=== Aug: 5-fold OOF, synthetic data in each fold's train set ==="
python train_cnn_aug_oof.py --data $DATA --ch 48 --epochs 80 \
    --select-by overall --tag aug --aug-samples $AUG
