#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 02:00:00
#SBATCH -J r144prep
#SBATCH -o logs/r144_prep_%j.out
# Clean-simulation analyses at 144 MHz, inputs: the flux-normalised
# ("shape-only") dataset and the relic catalogue from the 144 MHz maps.
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_shape_dataset.py \
    --input dataset_nh4_144_128.h5 --output dataset_nh4_144_128_shape.h5
./venv/bin/python -u detect_relics_radio.py --dataset dataset_nh4_144_512.h5 \
    --output relic_catalog_radio_nh4_144.h5
