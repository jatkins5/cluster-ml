#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J compare_dist
#SBATCH -o logs/compare_dist_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python compare_sim_obs_distributions.py
