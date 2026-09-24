#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=32G -t 00:30:00
#SBATCH -J cmpxray
#SBATCH -o logs/compare_xray_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_xray.py
