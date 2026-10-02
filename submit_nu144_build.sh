#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 16:00:00
#SBATCH -J nu144
#SBATCH -o logs/nu144_build_%j.out
# Emit the radio cells at 144 MHz instead of 1.4 GHz (Chuiyang's point):
# HB07 / Lee eq. 9 carries (nu / 1.4 GHz)^(-s/2) with s = s(Mach), so the
# 1.4 GHz -> 144 MHz rescaling differs cell by cell. A uniform rescaling is
# absorbed by the sim-flux offset; only this differential part matters.
# Same density cut (n_H < 1e-4) and centring as the nh4 baseline; the only
# change is the frequency, so every comparison against nh4 is paired.
cd /oscar/data/idellant/cluster-ml
set -e

./venv/bin/python -u build_radio_cells.py --max-nh 1e-4 --nu-ghz 0.144 --w-scale 12.5 \
    --out-dir Radio_Cells_nh4_144

for S in 512 128; do
  ./venv/bin/python -u build_dataset.py --center grouppos \
      --cells-dir Radio_Cells_nh4_144 --img-size $S \
      --output dataset_nh4_144_${S}.h5
done

./venv/bin/python -u build_injected_dataset.py \
    --dataset dataset_nh4_144_512.h5 --field-dir lotss_fields_40 \
    --flux-mode sim --output injected_nh4_144_simflux.h5

./venv/bin/python -u compare_nu144.py
