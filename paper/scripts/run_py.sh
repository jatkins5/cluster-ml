#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=32G -t 02:00:00
#SBATCH -J paperfig
#SBATCH -o logs/paperfig_%j.out
# Usage: sbatch paper/scripts/run_py.sh paper/scripts/<script>.py [args]
cd /oscar/data/idellant/cluster-ml
./venv/bin/python "$@"
