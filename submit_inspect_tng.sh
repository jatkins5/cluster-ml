#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J inspect_tng
#SBATCH -o logs/inspect_tng_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u inspect_tng_merger_cat.py
