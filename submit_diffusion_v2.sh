#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/diff_v2_%j.out
#SBATCH --error=logs/diff_v2_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Sharpness tune: relax the bright-pixel clip p99.9 -> p99.99 so compact
# bright knots (high-k structure) survive into training. Same arcsinh knee.
# Baseline (p99.9) gave normalized low-k 1.20x / DC 0.91x; this run tests
# whether the sharper-knot tune holds large-scale fidelity while reducing
# the mild over-smoothing visible in log10-PSD (0.10x).

python build_diffusion_data.py \
    --img-size 64 --extent-r500 2.0 \
    --hi-pct 99.99 \
    --output diffusion_radio_64_v2.h5

python train_diffusion.py \
    --data diffusion_radio_64_v2.h5 \
    --out-dir diffusion_out_v2 \
    --epochs 400 \
    --batch-size 64
