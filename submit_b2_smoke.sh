#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=48G -t 00:40:00
#SBATCH -J b2smoke
#SBATCH -o logs/b2_smoke_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
echo "=== compile"
./venv/bin/python -m py_compile train_cnn_mock.py train_cnn_pooled.py
echo "=== image only (2 folds x 3 epochs)"
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4.h5 \
    --folds 2 --epochs 3 --out-prefix /tmp/b2_img
echo "=== image + mass"
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4.h5 \
    --folds 2 --epochs 3 --mass --out-prefix /tmp/b2_mass
echo "=== image + blurred mass"
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4.h5 \
    --folds 2 --epochs 3 --mass --mass-noise --out-prefix /tmp/b2_massn
echo "=== mass only"
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4.h5 \
    --folds 2 --epochs 3 --mass --no-image --out-prefix /tmp/b2_massonly
echo "=== all smoke tests passed"
