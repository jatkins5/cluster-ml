#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 02:00:00
#SBATCH -J phasecut
#SBATCH -o logs/phase_cuts_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_phase_cuts.py
