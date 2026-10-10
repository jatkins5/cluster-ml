#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 08:00:00
#SBATCH -J f7build
#SBATCH -o logs/f7_build_%j.out
# Review F7: rebuild the snapshot-99 maps at 144 MHz WITHOUT the density cut
# (same w-scale, centring and grid as dataset_nh4_144_128), so the cut can be
# re-tested under the current protocol.
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -u build_radio_cells.py --nu-ghz 0.144 --w-scale 12.5 \
    --out-dir Radio_Cells_144
./venv/bin/python -u build_dataset.py --center grouppos --cells-dir Radio_Cells_144 \
    --img-size 128 --output dataset_nocut_144_128.h5
