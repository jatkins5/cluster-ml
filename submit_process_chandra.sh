#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 06:00:00
#SBATCH -J chproc
#SBATCH -o logs/process_chandra_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u process_chandra.py
