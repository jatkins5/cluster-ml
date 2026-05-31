#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-cond-cfg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/diff_cond_cfg_%j.out
#SBATCH --error=logs/diff_cond_cfg_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Sanity check: re-sample from the existing conditional EMA at higher CFG.
# Baseline cfg=1.5 saw aggregate-stat conditioning (corr=-0.46 brightness) but
# zero per-sample selectivity (every gen TSC bin's NN-train-TSC peaked at the
# same ~3 Gyr bulk). If the conditioning info is in the model and just being
# under-amplified, higher CFG should sharpen morphological selectivity. If
# not, the architecture is the limit and we move to AdaGN.

for CFG in 3.0 5.0; do
    TAG=cfg${CFG%.*}                                # cfg3, cfg5
    echo "=== sample-only at CFG=${CFG} (tag=${TAG}) ==="

    python train_diffusion.py \
        --data diffusion_radio_64_v2.h5 \
        --out-dir diffusion_out_cond \
        --condition \
        --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
        --label-key tsc_gyr --cond-scale-norm 8.0 \
        --sample-only --out-tag $TAG \
        --cfg-scale $CFG \
        --sample-tsc 0.3 1.0 2.0 4.0 6.0 --n-per-tsc 16

    python plot_cond_nn_check.py \
        --samples diffusion_out_cond/samples_cond_${TAG}.npz \
        --out-nn diffusion_out_cond/nn_check_cond_${TAG}.png \
        --out-leak diffusion_out_cond/cond_leakage_${TAG}.png
done
