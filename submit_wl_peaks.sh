#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J wlpeaks
#SBATCH -o logs/wl_peaks_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u analyze_wl_peaks.py
