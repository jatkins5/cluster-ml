#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 -N 1 -n 1 -c 4 --mem=32G -t 00:40:00
#SBATCH -J asmoke
#SBATCH -o logs/a_smoke_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
echo "=== compile"
./venv/bin/python -m py_compile train_cnn.py train_cnn_pooled.py train_cnn_mock.py
echo "=== pooled, inner selection (2 folds x 3 epochs)"
./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_128.h5 \
    --pseudo-tsc --folds 2 --epochs 3 --save-preds /tmp/smoke_pooled.npz
echo "=== pooled, final-epoch selection"
./venv/bin/python -u train_cnn_pooled.py --dataset dataset_nh4_128.h5 \
    --pseudo-tsc --folds 2 --epochs 3 --select final
echo "=== shallow, inner selection"
./venv/bin/python -u train_cnn.py --dataset dataset_nh4_128.h5 \
    --pseudo-tsc --folds 2 --epochs 3
echo "=== mock transfer, inner selection"
./venv/bin/python -u train_cnn_mock.py --dataset injected_nh4.h5 \
    --folds 2 --epochs 3 --out-prefix /tmp/smoke_inj
echo "=== all smoke tests passed"
