#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J camels_labels
#SBATCH -o logs/camels_labels_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python camels_labels.py
