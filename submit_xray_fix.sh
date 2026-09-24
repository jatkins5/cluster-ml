#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 00:30:00
#SBATCH -J xrayfix
#SBATCH -o logs/xray_fix_%j.out
# Two X-ray datasets with labels attached, identical apart from the stretch:
# the original (identity at these count rates) and the scaled fix.
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -m py_compile build_xray_dataset.py
./venv/bin/python -u build_xray_dataset.py --radio-dataset dataset_nh4_128.h5 \
    --restretch-from dataset_xray_128.h5 --stretch arcsinh \
    --labels-from dataset_nh4_128.h5 --output dataset_xray_128_orig.h5
./venv/bin/python -u build_xray_dataset.py --radio-dataset dataset_nh4_128.h5 \
    --restretch-from dataset_xray_128.h5 --stretch scaled \
    --labels-from dataset_nh4_128.h5 --output dataset_xray_128_scaled.h5
