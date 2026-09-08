#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J cnoise
#SBATCH -o logs/cnoise_%j.out
cd /oscar/data/idellant/cluster-ml
set -e

./venv/bin/python -u build_mock_dataset.py --dataset dataset_nh4_512.h5 \
    --correlated-noise --output mock_dataset_nh4_cn.h5

echo "############ domain gap, correlated noise ############"
./venv/bin/python -u check_domain_gap.py --dataset mock_dataset_nh4_cn.h5

echo "############ forward model MF, correlated noise ############"
./venv/bin/python -u forward_model_lotss.py --dataset dataset_nh4_512.h5 \
    --mask-compact --mask-max-beams 10 --correlated-noise \
    --out-prefix forward_nh4_cn
