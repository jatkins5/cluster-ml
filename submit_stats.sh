#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J rstats
#SBATCH -o logs/rstats_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_recenter_stats.py
