#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J synguard
#SBATCH -o logs/syn_guards_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u test_syn_guards.py
