#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:30:00
#SBATCH -J chprobe
#SBATCH -o logs/chandra_probe_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_chandra.py --probe
