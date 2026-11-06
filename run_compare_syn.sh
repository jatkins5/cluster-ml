#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=16G -t 01:30:00
#SBATCH -J cmpsyn
#SBATCH -o logs/compare_syn_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_syn.py
