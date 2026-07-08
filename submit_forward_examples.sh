#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J fwd_gallery
#SBATCH -o logs/forward_examples_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python plot_forward_examples.py
