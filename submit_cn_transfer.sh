#!/bin/bash
#SBATCH --job-name=cntrans
#SBATCH --partition=gpu
#SBATCH -N 1 -n 1 -c 4 --mem=24G --gres=gpu:1 -t 04:00:00
#SBATCH -o logs/cn_transfer_%j.out
#SBATCH -e logs/cn_transfer_%j.err
mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate
python train_cnn_mock.py --dataset mock_dataset_nh4_cn.h5 \
    --folds 5 --epochs 60 --out-prefix mocktransfer_nh4_cn
