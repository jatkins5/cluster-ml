#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -J recheck
#SBATCH -o logs/recenter_check_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_recenter.py
