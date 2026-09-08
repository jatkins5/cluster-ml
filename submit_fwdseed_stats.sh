#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J fwdstats
#SBATCH -o logs/fwdseed_stats_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_fwd_seed_stats.py
