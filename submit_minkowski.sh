#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J minkowski
#SBATCH -o logs/minkowski_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python minkowski_functionals.py
