#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=64G -t 02:00:00
#SBATCH -J jointreal
#SBATCH -a 0-2
#SBATCH -o logs/jointreal_%A_%a.out
# Joint radio + X-ray with both modalities observationally degraded: radio
# injected into real LoTSS fields with the simulation's own flux, X-ray
# thinned to real Chandra archive depths with sky background. Seeds match
# b2_sfimg_s42-44 (radio alone) and xreal_archive_s42-44 (X-ray alone).
cd /oscar/data/idellant/cluster-ml
SEED=$((42 + SLURM_ARRAY_TASK_ID))
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_simflux.h5 \
    --xray-dataset xray_real_archive.h5 --seed "$SEED" \
    --out-prefix "cnn_preds/jointreal_s${SEED}"
