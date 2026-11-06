#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J pvt
#SBATCH -o logs/pseudo_vs_tsc_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_pseudo_vs_tsc.py
