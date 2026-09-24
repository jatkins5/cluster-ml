#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 00:20:00
#SBATCH -J sumxd
#SBATCH -o logs/summarize_xdepth_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u summarize_xray_depth.py
