#!/bin/bash
#SBATCH --job-name=cluster-ml-cnn-aug-128
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_aug_128_%j.out
#SBATCH --error=logs/cnn_aug_128_%j.err

mkdir -p logs cnn_aug_out_128

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Phase 2 v3 step 2: CNN augmentation experiment at 128px (AdaGN samples).
# Uses the conditional samples from diffusion_out_cond_128_ada/samples_cond.npz
# (540 samples in TSC [0.3, 2.0] at cfg=5, produced by the AdaGN 128px
# diffusion model where conditioning genuinely works per-sample).
# Same cluster-level val split (--split-seed 0) across all CNN runs as at
# 64px, and identical to the diffusion model's split, so val clusters are
# real held-out throughout.
#
# Capacity scaled with resolution: ch=48 (1.38M params) since 128px gives
# more spatial info to exploit. Three seeds per condition.

OUTDIR=cnn_aug_out_128

for SEED in 0 1 2; do
    echo "=== baseline CNN @ 128px  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag baseline_s${SEED} \
        --data diffusion_radio_128_v2.h5 --ch 48 \
        --epochs 80 --batch-size 64
    mv cnn_aug_out/preds_baseline_s${SEED}.npz ${OUTDIR}/
done

for SEED in 0 1 2; do
    echo "=== augmented CNN @ 128px  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag aug_s${SEED} \
        --data diffusion_radio_128_v2.h5 --ch 48 \
        --aug-samples diffusion_out_cond_128_ada/samples_cond.npz \
        --epochs 80 --batch-size 64
    mv cnn_aug_out/preds_aug_s${SEED}.npz ${OUTDIR}/
done

python compare_cnn_aug.py --dir ${OUTDIR}
