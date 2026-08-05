#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J val_proxy
#SBATCH -o logs/val_proxy_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u validate_merger_proxy.py --stage "${1:-truth}"
