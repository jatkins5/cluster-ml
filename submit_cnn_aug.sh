#!/bin/bash
#SBATCH --job-name=cluster-ml-cnn-aug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --output=logs/cnn_aug_%j.out
#SBATCH --error=logs/cnn_aug_%j.err

mkdir -p logs cnn_aug_out

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Phase 2 augmentation experiment.
# Step 1: sample ~540 synthetic (image, TSC) pairs at CFG=5, restricted to
#         the TSC band where conditioning is clean (0.3-2.0 Gyr per the
#         CFG sweep diagnostic).
# Step 2: baseline CNN x 3 seeds (real data only)
# Step 3: augmented CNN x 3 seeds (real + synthetic)
# Step 4: compare predictions side-by-side
# Same cluster-level val split (--split-seed 0) across all CNN runs so the
# baseline-vs-aug comparison is on identical held-out clusters.

echo "=== step 1: generate augmentation samples (CFG=5, TSC 0.3-2.0) ==="
python train_diffusion.py \
    --data diffusion_radio_64_v2.h5 \
    --out-dir diffusion_out_cond \
    --condition \
    --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
    --label-key tsc_gyr --cond-scale-norm 8.0 \
    --sample-only --out-tag aug \
    --cfg-scale 5.0 \
    --sample-tsc 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 2.0 \
    --n-per-tsc 60

for SEED in 0 1 2; do
    echo "=== step 2: baseline CNN  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag baseline_s${SEED}
done

for SEED in 0 1 2; do
    echo "=== step 3: augmented CNN  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag aug_s${SEED} \
        --aug-samples diffusion_out_cond/samples_cond_aug.npz
done

echo "=== step 4: compare ==="
python compare_cnn_aug.py
