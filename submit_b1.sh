#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:20:00
#SBATCH -J b1mass
#SBATCH -o logs/b1_mass_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u analyze_mass_control.py
