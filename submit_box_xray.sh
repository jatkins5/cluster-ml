#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 04:00:00
#SBATCH -J boxxray
#SBATCH -o logs/box_xray_%j.out
# X-ray mocks re-placed on the 2 and 3 Mpc radio sets' redshift draws (26
# real targets now, so the draws changed), with the recalibrated brightness
# (L_X scale 0.358), both snapshots, merged, real Chandra inputs attached.
cd /oscar/data/idellant/cluster-ml
set -e
PY=./venv/bin/python
LX=0.358
for B in 2000 3000; do
  $PY -u build_xray_realistic.py build --z-from injected_nh4_144_b${B}_simflux.h5 \
      --lx-scale $LX --nh-csv lofar_nh.csv --output xray_placed_144_b${B}_lxcal.h5
  A=$($PY -c "import h5py; print(h5py.File('xray_placed_144_b${B}_lxcal.h5').attrs['stretch_scale'])")
  $PY -u build_xray_realistic.py build --counts-h5 xray_counts_2Ms_128_s91.h5 \
      --z-from injected_nh4_144_s91_b${B}_simflux.h5 --lx-scale $LX \
      --nh-csv lofar_nh.csv --stretch-scale "$A" --seed 91 \
      --output xray_placed_144_s91_b${B}_lxcal.h5
  $PY -u merge_snapshots.py xray_placed_144_b${B}_lxcal.h5 \
      xray_placed_144_s91_b${B}_lxcal.h5 --output xray_placed_144_s9991_b${B}_lxcal.h5
  $PY -u attach_xray_obs.py --obs-h5 xray_obs_acisi_pbc.h5 \
      --train-h5 xray_placed_144_s9991_b${B}_lxcal.h5
done
