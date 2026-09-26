#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 02:00:00
#SBATCH -J lxbuild
#SBATCH -o logs/lx_build_%j.out
# Redshift-placed X-ray mocks with TNG's L_X excess removed (1/3.6, from the
# catalogue comparison, not from the 15 clusters we are about to predict on)
# and each target's Galactic absorption applied; then the real Chandra
# inputs attached on the new set's stretch.
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_xray_realistic.py build \
    --z-from injected_nh4_simflux.h5 --lx-scale 0.2778 --nh-csv lofar_nh.csv \
    --output xray_real_placed_lx.h5
./venv/bin/python -u attach_xray_obs.py --train-h5 xray_real_placed_lx.h5
