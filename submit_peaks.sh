#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J peaks
#SBATCH -o logs/peaks_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u read_peak_catalogue.py
