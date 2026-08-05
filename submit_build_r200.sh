#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J build_r200
#SBATCH -o logs/build_r200_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_dataset.py --img-size 128 --extent-r200 2.0 \
    --output dataset_128_r200.h5
