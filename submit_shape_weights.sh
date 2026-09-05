#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J shape_weights
#SBATCH -o logs/shape_weights_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_sim_obs_distributions.py --shape-only \
    --out-prefix sim_obs_shape
