#!/bin/bash
#SBATCH --job-name=cnn-camproxy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_camproxy_%j.out
#SBATCH --error=logs/cnn_camproxy_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
OUT=cnn_camproxy_128
C="--camels dataset_camels_128.h5 --camels-labels camels_proxy_tsc.hdf5"
CLIP="--camels-tsc-clip 7.73"

# Baselines to beat, from the mass-jump label: pooled +0.416, transfer -0.437,
# TNG-only +0.435.
echo "### pooled: TNG + CAMELS with the mass-ratio proxy label ###"
$PY -u train_cnn_camels.py $C $CLIP --mode pooled \
    --tag proxy_pooled --out-dir $OUT --epochs 80

echo
echo "### pooled, cluster-scale zooms only (tests the group-vs-cluster gap) ###"
$PY -u train_cnn_camels.py $C $CLIP --camels-m200-min 1e14 --mode pooled \
    --tag proxy_pooled_m14 --out-dir $OUT --epochs 80

echo
echo "### transfer: CAMELS-trained, scored zero-shot on all TNG ###"
$PY -u train_cnn_camels.py $C $CLIP --mode transfer \
    --tag proxy_transfer --out-dir $OUT --epochs 80

echo
echo "### finetune: CAMELS pretrain then per-fold TNG tuning ###"
$PY -u train_cnn_camels.py $C $CLIP --mode finetune \
    --tag proxy_finetune --out-dir $OUT --epochs 80
