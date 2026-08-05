#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J inspect_camels
#SBATCH -o logs/inspect_camels_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python inspect_camels.py
