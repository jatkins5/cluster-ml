#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=16G -t 01:00:00
#SBATCH -J radiocmp
#SBATCH -o logs/compare_radio_versions_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_radio_versions.py
