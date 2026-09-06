#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 04:00:00
#SBATCH -J bldcells
#SBATCH -o logs/build_cells_%j.out

cd /oscar/data/idellant/cluster-ml

# 512px feeds the LoTSS forward model; 128px feeds the CNN baselines.
# Both GroupPos-centred, so cell smoothing is the only variable against
# dataset_gp_*.h5.
echo "=============== cells, 512px ==============="
./venv/bin/python -u build_dataset.py --center grouppos \
    --cells-dir Radio_Cells --img-size 512 --output dataset_cells_512.h5 || exit 1

echo "=============== point deposit, 512px (control) ==============="
./venv/bin/python -u build_dataset.py --center grouppos \
    --img-size 512 --output dataset_gp_512.h5 || exit 1

echo "=============== cells, 128px ==============="
./venv/bin/python -u build_dataset.py --center grouppos \
    --cells-dir Radio_Cells --img-size 128 --output dataset_cells_128.h5 || exit 1
