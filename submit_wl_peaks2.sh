#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J wlpeaks2
#SBATCH -o logs/wl_peaks2_%j.out
# Image-only predictions cover all 25 targets rather than the 17 with
# masses, which is the largest overlap with the peak catalogue available.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u analyze_wl_peaks.py \
    --preds "cnn_preds/b2_img_s*_preds.npz" --sn-min 3.0 4.0
