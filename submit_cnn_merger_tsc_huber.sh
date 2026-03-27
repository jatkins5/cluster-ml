#!/bin/bash
#SBATCH --job-name=cnn-mtsc-huber
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_merger_tsc_huber%a_%j.out
#SBATCH --error=logs/cnn_merger_tsc_huber%a_%j.err
#SBATCH --array=0-1

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

DELTAS=(1.0 2.0)
DELTA=${DELTAS[$SLURM_ARRAY_TASK_ID]}

echo "Running with Huber delta=$DELTA"

python train_cnn_pooled.py \
    --folds 5 \
    --epochs 60 \
    --batch-size 32 \
    --merger-tsc \
    --huber-delta $DELTA
