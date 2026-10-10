#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 04:00:00
#SBATCH -J xdepthb
#SBATCH -o logs/xdepth_fix_build_%j.out
# X-ray depth study redone under the corrected count model (review F5):
# every mock at a fixed Cycle-22-equivalent exposure, livetime from its
# target's ratio, snapshot 99, same placements as xray_fix.h5.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
A=$(./venv/bin/python -c "import h5py; print(h5py.File('xray_fix.h5').attrs['stretch_scale'])")
for KS in 5 20 50 150 500; do
  ./venv/bin/python -u build_xray_realistic.py build --count-model real --fixed-ks $KS \
      --z-from injected_fix_phys_b1000.h5 --lx-scale 0.358 --nh-csv lofar_nh.csv \
      --stretch-scale "$A" --output xray_depthfix_${KS}ks.h5
done
