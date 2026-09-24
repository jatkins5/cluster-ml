#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 03:00:00
#SBATCH -J xrplace
#SBATCH -o logs/xray_placed_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -m py_compile build_xray_realistic.py
./venv/bin/python -u build_xray_realistic.py pbkg --n-bkg 4
./venv/bin/python -u build_xray_realistic.py build \
    --z-from injected_nh4_simflux.h5 --output xray_real_placed.h5
