#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 16:00:00
#SBATCH -J synnocut
#SBATCH -o logs/syn_nocut_%j.out
# Review F7 on the regenerated radio model: the same snapshot-99 maps WITHOUT
# the n_H < 1e-4 cut, to retest whether the cut still matters (paper: no
# accuracy gain, p = 0.16, under the old weights).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
C=/oscar/data/idellant/Chuiyang
./venv/bin/python -u build_radio_cells.py --model synch --nu-ghz 0.144 --w-scale 206 \
    --radio-dir $C/Radio_Snap99_new --cutout-dir $C/TNGCluster_Cutout/snap99 \
    --out-dir Radio_Cells_syn_144
./venv/bin/python -u build_dataset.py --center grouppos --cells-dir Radio_Cells_syn_144 \
    --img-size 128 --output dataset_syn_nocut_144_128.h5
