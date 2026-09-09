#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J inject
#SBATCH -o logs/inject_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_injected_dataset.py --dataset dataset_nh4_512.h5 \
    --output injected_nh4.h5
echo "############ domain gap, field-injected ############"
./venv/bin/python -u check_domain_gap.py --dataset injected_nh4.h5
