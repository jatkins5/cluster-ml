#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J survivors
#SBATCH -o logs/survivors_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_surviving_sources.py
