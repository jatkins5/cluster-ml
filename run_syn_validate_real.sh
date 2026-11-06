#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=8G -t 00:20:00
#SBATCH -J synval
#SBATCH -o logs/syn_validate_real_%j.out
# Validators only (no bootstrap, no scientific analysis) on EVERY declared
# comparison of compare_syn.py (shared list).
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_syn.py --validate-only
