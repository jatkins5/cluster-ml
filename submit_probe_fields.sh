#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:30:00
#SBATCH -J probefld
#SBATCH -o logs/probe_fields_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u probe_lotss_fields.py
