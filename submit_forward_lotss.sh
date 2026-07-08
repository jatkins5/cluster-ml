#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J fwd_lotss
#SBATCH -o logs/forward_lotss_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python forward_model_lotss.py
