#!/bin/bash
#SBATCH --job-name=cluster-ml-cnn-aug-128-cfg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_aug_128_cfg_%j.out
#SBATCH --error=logs/cnn_aug_128_cfg_%j.err

mkdir -p logs cnn_aug_out_128_cfg3 cnn_aug_out_128_cfg15

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Re-run CNN augmentation at 128px with lower-CFG samples that have less
# distribution shift, to test whether the methodology can shift the
# significant aug benefit from late-merger (the CFG=5 result) toward
# recent-merger (the original goal).
#
# Samples already saved from the diffusion-side CFG sweep:
#   diffusion_out_cond_128_ada/samples_cond_cfg3.npz   (low-k 2.98x, looser conditioning)
#   diffusion_out_cond_128_ada/samples_cond_cfg15.npz  (low-k 2.82x, loosest)
# No new diffusion sampling needed. Cluster-level val split fixed at
# --split-seed 0 (matches the CFG=5 run), so the comparison to that
# experiment's baseline is on identical val clusters.

# CFG=3 aug -> cnn_aug_out_128_cfg3/
for SEED in 0 1 2; do
    echo "=== CFG=3 aug CNN @ 128px  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag aug_s${SEED} \
        --out-dir cnn_aug_out_128_cfg3 \
        --data diffusion_radio_128_v2.h5 --ch 48 \
        --aug-samples diffusion_out_cond_128_ada/samples_cond_cfg3.npz \
        --epochs 80 --batch-size 64
done

# CFG=1.5 aug -> cnn_aug_out_128_cfg15/
for SEED in 0 1 2; do
    echo "=== CFG=1.5 aug CNN @ 128px  seed=$SEED ==="
    python train_cnn_aug.py --seed $SEED --tag aug_s${SEED} \
        --out-dir cnn_aug_out_128_cfg15 \
        --data diffusion_radio_128_v2.h5 --ch 48 \
        --aug-samples diffusion_out_cond_128_ada/samples_cond_cfg15.npz \
        --epochs 80 --batch-size 64
done

# Compare each against the CFG=5 baseline that's already in cnn_aug_out_128/.
# (Baseline runs are independent of which aug samples are used, so we don't
# need to retrain them - just symlink for the per-dir compare.)
for D in cnn_aug_out_128_cfg3 cnn_aug_out_128_cfg15; do
    for SEED in 0 1 2; do
        ln -sf ../cnn_aug_out_128/preds_baseline_s${SEED}.npz \
            ${D}/preds_baseline_s${SEED}.npz
    done
    python compare_cnn_aug.py --dir ${D}
done
