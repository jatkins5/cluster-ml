#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J centering
#SBATCH -o logs/centering_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_centering.py
