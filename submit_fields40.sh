#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=8G -t 02:00:00
#SBATCH -J fields40
#SBATCH -o logs/fields40_%j.out
# The background fields have to be at least as large as the biggest box any
# injection redshift implies. Recovering A2052/A2063/A2147 (z=0.035) means a
# 1 Mpc box now subtends ~23 arcmin, so the 20 arcmin fields fail the crop
# and a third of clusters were being dropped. Same seed, so the sky
# positions are the same ones as the 20 arcmin set.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_lotss_fields.py \
    --out-dir lotss_fields_40 --size-arcmin 40 --seed 0
