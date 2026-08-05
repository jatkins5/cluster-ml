#!/bin/bash
#SBATCH --job-name=cnn-proxy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/cnn_proxy_%j.out
#SBATCH --error=logs/cnn_proxy_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
OUT=cnn_proxy_128

$PY -u build_proxy_tsc.py || exit 1

echo
echo "### trained on proxy TSC, censored halos filled at the catalog span ###"
$PY -u train_cnn_camels.py --labels tsc_proxy_snap99.hdf5 \
    --label-key tsc_proxy --tag proxy --out-dir $OUT --epochs 80

echo
echo "### trained on proxy TSC, uncensored halos only ###"
$PY -u train_cnn_camels.py --labels tsc_proxy_snap99.hdf5 \
    --label-key tsc_proxy_unc --tag proxy_unc --out-dir $OUT --epochs 80

echo
echo "### rescored against the TRUE TSC ###"
$PY -u score_vs_truth.py
