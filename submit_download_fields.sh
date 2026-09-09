#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 08:00:00
#SBATCH -J dlfields
#SBATCH -o logs/download_fields_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u download_lotss_fields.py --restrict-to-local
