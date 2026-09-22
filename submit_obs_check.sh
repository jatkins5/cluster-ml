#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:20:00
#SBATCH -J obschk
#SBATCH -o logs/obs_check_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_obs_sample.py
