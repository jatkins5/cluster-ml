#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-cond-128
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/diff_cond_128_%j.out
#SBATCH --error=logs/diff_cond_128_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Phase 2 v2 step 1: conditional diffusion at 128px.
# Same architecture and conditioning as the 64px run (clip p99.99,
# Fourier scalar condition + null token, CFG dropout 0.1), just more
# pixels. batch_size halved to 32 since activations scale ~4x. epochs
# reduced to 300 (64px loss plateaued by ~100; 300 gives headroom).
# Sample at the wide TSC grid we'll use for CNN augmentation, at CFG=5
# (the operating point that worked at 64px), so the end-of-training
# samples are directly reusable as augmentation data.

echo "=== step 1: build diffusion_radio_128_v2.h5 ==="
python build_diffusion_data.py --img-size 128 --hi-pct 99.99 \
    --output diffusion_radio_128_v2.h5

echo "=== step 2: train conditional diffusion at 128px ==="
python train_diffusion.py \
    --data diffusion_radio_128_v2.h5 \
    --out-dir diffusion_out_cond_128 \
    --epochs 300 --batch-size 32 \
    --condition \
    --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
    --label-key tsc_gyr --cond-scale-norm 8.0 \
    --cond-drop-prob 0.1 --cfg-scale 5.0 \
    --sample-tsc 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 2.0 \
    --n-per-tsc 60

echo "=== step 3: per-bin NN diagnostic ==="
python plot_cond_nn_check.py \
    --samples diffusion_out_cond_128/samples_cond.npz \
    --data diffusion_radio_128_v2.h5 \
    --out-nn diffusion_out_cond_128/nn_check_cond.png \
    --out-leak diffusion_out_cond_128/cond_leakage.png
