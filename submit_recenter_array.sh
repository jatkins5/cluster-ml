#!/bin/bash
#SBATCH --job-name=recenter-arr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/recenter_arr_%A_%a.out
#SBATCH --error=logs/recenter_arr_%A_%a.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# One (model, dataset, seed) per array task so the eight configurations run
# side by side. Two seeds because the run-to-run OOF R2 noise floor is ~0.012.
MODELS=(train_cnn.py train_cnn_pooled.py)
DATASETS=(dataset_wc_128.h5 dataset_gp_128.h5)
SEEDS=(42 43)

I=$SLURM_ARRAY_TASK_ID
M=${MODELS[$(( I % 2 ))]}
D=${DATASETS[$(( (I / 2) % 2 ))]}
S=${SEEDS[$(( I / 4 ))]}

echo "=========== $M  $D  seed=$S ==========="
python "$M" --folds 5 --epochs 60 --batch-size 32 \
    --seed "$S" --pseudo-tsc --dataset "$D"
