#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 01:00:00
#SBATCH -J emitgas
#SBATCH -o logs/emitting_gas_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_emitting_gas.py
