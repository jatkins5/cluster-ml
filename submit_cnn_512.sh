#!/bin/bash
#SBATCH --job-name=cnn-512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/cnn_512_%j.out
#SBATCH --error=logs/cnn_512_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== Building 512px dataset ==="
python build_dataset.py --img-size 512 --output dataset_512.h5

echo ""
echo "=== Training + plotting ==="
python plot_tsc_predictions.py --dataset dataset_512.h5 --img-size 512 --output tsc_true_vs_pred_512.png --batch-size 8
