#!/bin/bash
#SBATCH --job-name=cnn-camels2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_camels2_%j.out
#SBATCH --error=logs/cnn_camels2_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python

echo "### 1. TRANSFER: CAMELS-trained -> all TNG ###"
$PY -u train_cnn_camels.py --mode transfer --tag transfer \
    --camels dataset_camels_128.h5 --camels-tsc-max 8.0 --epochs 80

echo
echo "### 2. FINETUNE: CAMELS pretrain -> per-fold TNG tuning ###"
$PY -u train_cnn_camels.py --mode finetune --tag finetune \
    --camels dataset_camels_128.h5 --camels-tsc-max 8.0 \
    --pretrain-epochs 80 --epochs 80 --finetune-lr 1e-4
