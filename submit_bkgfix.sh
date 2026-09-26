#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 03:00:00
#SBATCH -J bkgfix
#SBATCH -o logs/bkgfix_%j.out
# Background fix on both sides: mocks get backgrounds scaled by the sky area
# a block covers at their redshift; real data get the particle background
# measured at 9.5-12 keV, subtracted, and replaced by the mocks' level, with
# chip gaps filled. Then the real inputs are attached on the mocks' stretch.
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_xray_realistic.py build \
    --z-from injected_nh4_simflux.h5 --lx-scale 0.2778 --nh-csv lofar_nh.csv \
    --output xray_real_placed_lx.h5
./venv/bin/python -u process_chandra.py --train-h5 xray_real_placed_lx.h5 \
    --output xray_obs_acisi_pbc.h5
./venv/bin/python -u attach_xray_obs.py --obs-h5 xray_obs_acisi_pbc.h5 \
    --train-h5 xray_real_placed_lx.h5
