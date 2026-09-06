#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J wrange
#SBATCH -o logs/weight_range_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_weight_range.py
