#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=8G -t 00:30:00
#SBATCH -J scoremse
#SBATCH -o logs/score_lbl_mse_%j.out
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_lbl_mse.py --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d
