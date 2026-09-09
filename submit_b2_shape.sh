#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=24G -t 00:30:00
#SBATCH -J shapeds
#SBATCH -o logs/shape_build_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_shape_dataset.py \
    --input dataset_nh4_128.h5 --output dataset_nh4_128_shape.h5
