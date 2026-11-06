#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 02:00:00
#SBATCH -J cyq
#SBATCH -o logs/chuiyang_q_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_chuiyang_q.py
