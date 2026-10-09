#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 08:00:00
#SBATCH -J figdata
#SBATCH -o logs/fig_data_%j.out
# Data for two new paper figures: the emitting-gas phase diagram (all 352
# snapshot-99 cutouts) and the X-ray depth study (redshift-placed mocks at
# fixed exposures, every correction applied).
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u extract_phase.py
for KS in 5 20 50 150 500; do
  ./venv/bin/python -u build_xray_realistic.py build \
      --z-from injected_nh4_144_simflux.h5 --lx-scale 0.2778 \
      --nh-csv lofar_nh.csv --fixed-ks $KS --output xray_depth_${KS}ks.h5
done
