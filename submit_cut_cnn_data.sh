#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 02:00:00
#SBATCH -J cutcnndata
#SBATCH -o logs/cut_cnn_data_%j.out

cd /oscar/data/idellant/cluster-ml
set -e
for TAG in nh3 nh4; do
    echo "=============== 128px dataset, cut $TAG ==============="
    ./venv/bin/python -u build_dataset.py --center grouppos \
        --cells-dir "Radio_Cells_$TAG" --img-size 128 \
        --output "dataset_${TAG}_128.h5"
done
