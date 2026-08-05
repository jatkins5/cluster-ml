#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J camels_mass
#SBATCH -o logs/camels_mass_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u camels_mass_tsc_check.py
