#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J audit
#SBATCH -o logs/audit_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u audit_results.py
./venv/bin/python -u audit_mass_confound.py
