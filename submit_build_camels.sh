#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J build_camels
#SBATCH -o logs/build_camels_%j.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_camels_dataset.py --img-size 128 \
    --extent-r200 2.0 --threshold 0.10 --use-rate --min-logm200 14.0 \
    --output dataset_camels_128.h5
