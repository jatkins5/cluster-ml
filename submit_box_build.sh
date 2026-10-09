#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 08:00:00
#SBATCH -J boxbuild
#SBATCH -o logs/box_build_%j.out
# Fix 1: injected radio mocks over 2 and 3 Mpc boxes instead of 1 Mpc (27%
# and 65% of a cluster's shock emission instead of 4%), both snapshots, from
# the 80' LoTSS targets and fields. The flux constant is refit per box at
# snapshot 99 and reused for 91.
cd /oscar/data/idellant/cluster-ml
set -e
PY=./venv/bin/python
for B in 2000 3000; do
  $PY -u build_injected_dataset.py --dataset dataset_nh4_144_512.h5 \
      --field-dir lotss_fields_80 --lotss-glob "lotss_images_80/*.fits" \
      --box-kpc $B --flux-mode sim --output injected_nh4_144_b${B}_simflux.h5
  OFF=$($PY -c "import h5py; print(h5py.File('injected_nh4_144_b${B}_simflux.h5').attrs['sim_flux_offset_dex'])")
  $PY -u build_injected_dataset.py --dataset dataset_nh4_144_s91_512.h5 \
      --field-dir lotss_fields_80 --lotss-glob "lotss_images_80/*.fits" \
      --box-kpc $B --flux-mode sim --sim-offset "$OFF" --seed 91 \
      --output injected_nh4_144_s91_b${B}_simflux.h5
  $PY -u merge_snapshots.py injected_nh4_144_b${B}_simflux.h5 \
      injected_nh4_144_s91_b${B}_simflux.h5 --output injected_nh4_144_s9991_b${B}_simflux.h5
  # the X-ray mocks are paired by redshift draw; say whether the 1 Mpc set's
  # draws carry over (same surviving targets -> same random sequence)
  $PY -c "
import h5py, numpy as np
a = h5py.File('injected_nh4_144_s9991_simflux.h5')['mock/z'][:]
b = h5py.File('injected_nh4_144_s9991_b${B}_simflux.h5')['mock/z'][:]
print('box $B: redshift draws identical to the 1 Mpc set:', a.shape == b.shape and np.array_equal(a, b))"
done
