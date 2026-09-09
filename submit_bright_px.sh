#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:30:00
#SBATCH -J brightpx
#SBATCH -o logs/bright_px_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_bright_pixels.py
