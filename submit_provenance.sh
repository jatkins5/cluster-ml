#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 02:00:00
#SBATCH -J prov
#SBATCH -o logs/provenance_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_obs_provenance.py
