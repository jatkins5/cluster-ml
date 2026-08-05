#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J major_tsc
#SBATCH -o logs/major_tsc_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_major_tsc.py
