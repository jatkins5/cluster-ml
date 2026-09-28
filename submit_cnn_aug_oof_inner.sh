#!/bin/bash
#SBATCH --job-name=cnn-aug-inner
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/cnn_aug_inner_%A_%a.out
#SBATCH --error=logs/cnn_aug_inner_%A_%a.err
# Synthetic-augmentation test redone under inner-split checkpoint selection:
# one paired (baseline, aug) run per seed; array task = seed.

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

DATA=diffusion_radio_128_v2.h5
AUG=diffusion_out_cond_128_ada/samples_cond.npz
S=$SLURM_ARRAY_TASK_ID
OUT=cnn_aug_oof_128_inner

python train_cnn_aug_oof.py --data $DATA --ch 48 --epochs 80 --select inner \
    --seed $S --tag baseline_s$S --out-dir $OUT
python train_cnn_aug_oof.py --data $DATA --ch 48 --epochs 80 --select inner \
    --seed $S --tag aug_s$S --out-dir $OUT --aug-samples $AUG
