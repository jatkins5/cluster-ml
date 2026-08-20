#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J cam_bench
#SBATCH -o logs/cam_bench_%j.out

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
time ./venv/bin/python -u build_camels_proxy_tsc.py --stage zooms \
    --start 0 --end 8 --force
