#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 00:20:00
#SBATCH -J jsmoke
#SBATCH -o logs/joint_smoke_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
./venv/bin/python -m py_compile train_cnn_pooled.py train_cnn_mock.py
./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_128.h5 \
    --xray-dataset dataset_xray_128_scaled.h5 --pseudo-tsc --folds 2 \
    --epochs 2 --save-preds /tmp/joint_smoke.npz
./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_128.h5 \
    --pseudo-tsc --folds 2 --epochs 2
echo "=== smoke passed"
