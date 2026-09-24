#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=64G -t 00:30:00
#SBATCH -J jrsmoke
#SBATCH -o logs/jointreal_smoke_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -m py_compile train_cnn_mock.py
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_simflux.h5 \
    --xray-dataset xray_real_archive.h5 --folds 2 --epochs 2 \
    --out-prefix /tmp/jr_smoke
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4_simflux.h5 \
    --folds 2 --epochs 2 --out-prefix /tmp/jr_smoke_radio
echo "=== smoke passed"
