#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 03:00:00
#SBATCH -J c3build
#SBATCH -o logs/c3_rebuild_%j.out
# C3: rebuild the field-injected dataset now that C1/C2 recovered the
# dropped cutouts -- the obs group goes from 17 to 25 targets.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_injected_dataset.py \
    --dataset dataset_nh4_512.h5 --output injected_nh4.h5
