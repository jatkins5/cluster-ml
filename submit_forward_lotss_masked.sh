#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J fwd_masked
#SBATCH -o logs/forward_lotss_masked_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u forward_model_lotss.py --mask-compact \
    --out-prefix forward_lotss_masked
