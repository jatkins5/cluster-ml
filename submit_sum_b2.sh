#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:15:00
#SBATCH -J sumb2
#SBATCH -o logs/summarize_b2_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u summarize_b2.py
