#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 01:00:00
#SBATCH -J chkxray
#SBATCH -o logs/check_xray_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_xray.py
