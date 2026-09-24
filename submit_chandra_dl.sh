#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 08:00:00
#SBATCH -J chdl
#SBATCH -o logs/chandra_dl_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_chandra.py
du -sh ~/data/cluster-ml/chandra
