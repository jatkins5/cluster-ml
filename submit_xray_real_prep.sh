#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 04:00:00
#SBATCH -J xrprep
#SBATCH -o logs/xray_real_prep_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_xray_realistic.py counts
./venv/bin/python -u build_xray_realistic.py bkg --n-bkg 6
