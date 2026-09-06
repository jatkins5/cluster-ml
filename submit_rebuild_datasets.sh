#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J rebuild
#SBATCH -o logs/rebuild_%j.out

cd /oscar/data/idellant/cluster-ml

echo "=================== GroupPos-centred, 128px ==================="
./venv/bin/python -u build_dataset.py --center grouppos \
    --img-size 128 --output dataset_gp_128.h5 || exit 1

echo "=================== legacy weight-centred, 128px (control) ==================="
./venv/bin/python -u build_dataset.py --center weight \
    --img-size 128 --output dataset_wc_128.h5 || exit 1
