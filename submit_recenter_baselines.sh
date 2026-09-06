#!/bin/bash
#SBATCH --job-name=recenter-base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/recenter_baselines_%j.out
#SBATCH --error=logs/recenter_baselines_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Two seeds per configuration: run-to-run noise on OOF R2 is ~0.012 (cuDNN
# conv nondeterminism), so a single pair of numbers cannot separate a real
# shift from scatter.
for SEED in 42 43; do
  for DS in dataset_wc_128.h5 dataset_gp_128.h5; do
    echo "=========== shallow  $DS  seed=$SEED ==========="
    python train_cnn.py --folds 5 --epochs 60 --batch-size 32 \
        --seed "$SEED" --pseudo-tsc --dataset "$DS"

    echo "=========== pooled   $DS  seed=$SEED ==========="
    python train_cnn_pooled.py --folds 5 --epochs 60 --batch-size 32 \
        --seed "$SEED" --pseudo-tsc --dataset "$DS"
  done
done
