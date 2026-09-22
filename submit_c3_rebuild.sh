#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 03:00:00
#SBATCH -J c3build
#SBATCH -o logs/c3_rebuild_%j.out
# C3: rebuild the field-injected dataset on the recovered sample. The obs
# group goes 17 -> 25 targets (C1/C2) and the backgrounds are the 40 arcmin
# set, which is what lets the z=0.035 targets be injected at all.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_injected_dataset.py \
    --dataset dataset_nh4_512.h5 --field-dir lotss_fields_40 \
    --output injected_nh4.h5
