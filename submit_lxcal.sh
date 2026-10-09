#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 04:00:00
#SBATCH -J lxcal
#SBATCH -o logs/lxcal_%j.out
# Fix 2 test: real clusters are 1.29x brighter in X-rays than mass-matched
# mocks after the catalogue L_X correction (fig_lx_mass), and every real
# X-ray prediction is younger than the radio one. Recalibrate the mocks'
# brightness by that factor (0.2778 x 1.29 = 0.358) and rebuild both
# snapshots on the same radio redshifts.
cd /oscar/data/idellant/cluster-ml
set -e
PY=./venv/bin/python
LX=0.358
$PY -u build_xray_realistic.py build --z-from injected_nh4_144_simflux.h5 \
    --lx-scale $LX --nh-csv lofar_nh.csv --output xray_placed_144_lxcal.h5
A=$($PY -c "import h5py; print(h5py.File('xray_placed_144_lxcal.h5').attrs['stretch_scale'])")
$PY -u build_xray_realistic.py build --counts-h5 xray_counts_2Ms_128_s91.h5 \
    --z-from injected_nh4_144_s91_simflux.h5 --lx-scale $LX --nh-csv lofar_nh.csv \
    --stretch-scale "$A" --seed 91 --output xray_placed_144_s91_lxcal.h5
$PY -u merge_snapshots.py xray_placed_144_lxcal.h5 xray_placed_144_s91_lxcal.h5 \
    --output xray_placed_144_s9991_lxcal.h5
$PY -u attach_xray_obs.py --obs-h5 xray_obs_acisi_pbc.h5 \
    --train-h5 xray_placed_144_s9991_lxcal.h5
