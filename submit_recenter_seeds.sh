#!/bin/bash
#SBATCH --job-name=recenter-seed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/recenter_seed_%A_%a.out
#SBATCH --error=logs/recenter_seed_%A_%a.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Seeds 44-46 on top of 42-43, because the wc-vs-gp deltas (+0.005 shallow,
# +0.018 pooled) are the same size as the seed-to-seed spread at n=2.
MODELS=(train_cnn.py train_cnn_pooled.py)
DATASETS=(dataset_wc_128.h5 dataset_gp_128.h5)
SEEDS=(44 45 46)

I=$SLURM_ARRAY_TASK_ID
M=${MODELS[$(( I % 2 ))]}
D=${DATASETS[$(( (I / 2) % 2 ))]}
S=${SEEDS[$(( I / 4 ))]}

echo "=========== $M  $D  seed=$S ==========="
python "$M" --folds 5 --epochs 60 --batch-size 32 \
    --seed "$S" --pseudo-tsc --dataset "$D"
