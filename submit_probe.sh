#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 00:30:00
#SBATCH -J probe
#SBATCH -o logs/probe_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u probe_cutout.py
