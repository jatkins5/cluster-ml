#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J bldmock
#SBATCH -o logs/build_mock_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
# nh4 is the configuration chosen for realism; nh3 as the ablation.
./venv/bin/python -u build_mock_dataset.py --dataset dataset_nh4_512.h5 \
    --output mock_dataset_nh4.h5
./venv/bin/python -u build_mock_dataset.py --dataset dataset_nh3_512.h5 \
    --output mock_dataset_nh3.h5
./venv/bin/python -u build_mock_dataset.py --dataset dataset_gp_512.h5 \
    --output mock_dataset_nocut.h5
