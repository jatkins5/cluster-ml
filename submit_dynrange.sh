#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -J dynrange
#SBATCH -o logs/dynrange_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_mock_dynamic_range.py
