#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 03:00:00
#SBATCH -J b23build
#SBATCH -o logs/b23_build_%j.out
# B2 option 3: drop the observed flux anchor. Each mock's total flux now
# comes from the simulation's own predicted power, with one global constant
# setting the absolute scale. Mass dependence and scatter become predictions
# rather than inputs.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_injected_dataset.py \
    --dataset dataset_nh4_512.h5 --field-dir lotss_fields_40 \
    --flux-mode sim --output injected_nh4_simflux.h5
