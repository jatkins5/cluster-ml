#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J wlmass
#SBATCH -o logs/wl_masses_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u collect_wl_masses.py
