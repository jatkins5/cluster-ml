#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J camels_gc
#SBATCH -o logs/camels_gc_%j.out

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u inspect_camels_groupcat.py
