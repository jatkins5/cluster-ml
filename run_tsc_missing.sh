#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:15:00
#SBATCH -J tscmiss
#SBATCH -o logs/tsc_missing_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_tsc_missing.py
