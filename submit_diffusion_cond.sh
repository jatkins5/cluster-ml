#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-cond
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/diff_cond_%j.out
#SBATCH --error=logs/diff_cond_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Conditional DDPM on merger-TSC. Reuses the v2 data (p99.99 clip) as that
# was the clean baseline. Conditioning adds ~100k params (cproj + learned
# null token). Standard CFG dropout 0.1, sample CFG scale 1.5.
#
# Gate question: does the gen morphology actually shift with the TSC
# condition? evaluate_conditional_response() reports the per-bin brightness
# and corr(condition, gen brightness); also dumps a TSC-rowed sample grid
# (cond_grid_final.png) for visual inspection.
#
# Sample at 5 TSC values spanning 0.3-6.0 Gyr (data range is 0-7.7),
# 16 samples each = 80 total.

python train_diffusion.py \
    --data diffusion_radio_64_v2.h5 \
    --out-dir diffusion_out_cond \
    --epochs 400 --batch-size 64 \
    --condition \
    --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
    --label-key tsc_gyr --cond-scale-norm 8.0 \
    --cond-drop-prob 0.1 --cfg-scale 1.5 \
    --sample-tsc 0.3 1.0 2.0 4.0 6.0 --n-per-tsc 16
