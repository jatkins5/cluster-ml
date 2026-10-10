#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 04:00:00
#SBATCH -J fixxbox
#SBATCH -o logs/fix_xray_boxes_%j.out
# The radio sets' redshift draws differ between box sizes (A1750 yields a
# usable cutout at some sizes and not others, changing the random sequence),
# so the joint models need X-ray mocks placed on each set's own draws. Same
# count model, calibration and stretch constant as xray_fix.h5.
cd /oscar/data/idellant/cluster-ml
set -e
PY="./venv/bin/python -u"
A=$($PY -c "import h5py; print(h5py.File('xray_fix.h5').attrs['stretch_scale'])")
for B in 2000 3000; do
  $PY build_xray_realistic.py build --count-model real \
      --z-from injected_fix_phys_b${B}.h5 --lx-scale 0.358 --nh-csv lofar_nh.csv \
      --stretch-scale "$A" --output xray_fix_b${B}.h5
  $PY build_xray_realistic.py build --count-model real --counts-h5 xray_counts_2Ms_128_s91.h5 \
      --z-from injected_fix_phys_b${B}_s91.h5 --lx-scale 0.358 --nh-csv lofar_nh.csv \
      --stretch-scale "$A" --seed 91 --output xray_fix_b${B}_s91.h5
  $PY merge_snapshots.py xray_fix_b${B}.h5 xray_fix_b${B}_s91.h5 --output xray_fix_b${B}_s9991.h5
  $PY attach_xray_obs.py --obs-h5 xray_obs_fix.h5 --train-h5 xray_fix_b${B}_s9991.h5
  $PY -c "
import h5py, numpy as np
r = h5py.File('injected_fix_phys_b${B}_s9991.h5')['mock/z'][:]
x = h5py.File('xray_fix_b${B}_s9991.h5')['mock/z'][:]
print('box ${B}: X-ray redshifts match the radio set:', np.allclose(r[:, :x.shape[1]], x))"
done
