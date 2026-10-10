#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 06:00:00
#SBATCH -J optpilot
#SBATCH -o logs/optical_pilot_%j.out
# E1/E2 idealised optical pilot: catalogue features (reads group catalogues
# and gas cutouts, read-only), then the tabular comparison. CPU only.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u optical_catalog.py
./venv/bin/python -u train_optical_pilot.py
