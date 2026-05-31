#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-cond-128-ada
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/diff_cond_128_ada_%j.out
#SBATCH --error=logs/diff_cond_128_ada_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Phase 2 v3: conditional diffusion at 128px with AdaGN.
# Same config as the failed 128px-no-ada run (--cond, --cfg-scale 5,
# 300 epochs, batch 32, TSC grid 0.3..2.0 at 60 each), with --ada added.
# AdaGN replaces plain GroupNorm with condition-modulated norms in every
# ResBlock, so the TSC condition shapes per-channel scale+shift at every
# depth rather than only nudging the time embedding once. Zero-init proj
# means training starts at the identity (same effective state as plain
# GroupNorm), so loss should still plateau cleanly; the conditioning
# pathway is what gains capacity.

echo "=== train AdaGN conditional diffusion at 128px ==="
python train_diffusion.py \
    --data diffusion_radio_128_v2.h5 \
    --out-dir diffusion_out_cond_128_ada \
    --epochs 300 --batch-size 32 \
    --condition --ada \
    --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
    --label-key tsc_gyr --cond-scale-norm 8.0 \
    --cond-drop-prob 0.1 --cfg-scale 5.0 \
    --sample-tsc 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 2.0 \
    --n-per-tsc 60

echo "=== per-bin NN diagnostic ==="
python plot_cond_nn_check.py \
    --samples diffusion_out_cond_128_ada/samples_cond.npz \
    --data diffusion_radio_128_v2.h5 \
    --out-nn diffusion_out_cond_128_ada/nn_check_cond.png \
    --out-leak diffusion_out_cond_128_ada/cond_leakage.png
