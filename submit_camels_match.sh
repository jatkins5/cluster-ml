#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J cam_mtch
#SBATCH -o logs/cam_mtch_%j.out

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_camels_label_match.py
