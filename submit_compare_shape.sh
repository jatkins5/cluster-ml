#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:20:00
#SBATCH -J cmpshape
#SBATCH -o logs/compare_shape_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u compare_shape.py
echo
echo "################ full mass-control readout, shape-only model"
./venv/bin/python -u analyze_mass_control.py \
    --preds "cnn_preds/pooled_shape_inner_s*.npz"
