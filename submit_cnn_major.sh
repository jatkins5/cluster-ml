#!/bin/bash
#SBATCH --job-name=cnn-major
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/cnn_major_%j.out
#SBATCH --error=logs/cnn_major_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python
L=tsc_major_snap99.hdf5
OUT=cnn_major_128

# Control: the shipped any-ratio label restricted to the same 255 halos, so
# a difference against the mr033 run is the label and not the sample.
echo "### control: any-ratio TSC on the major-merger subset ###"
$PY -u train_cnn_camels.py --labels $L --label-key tsc_any_on_major \
    --tag any_on_major --out-dir $OUT --epochs 80

echo
echo "### major merger TSC (mass ratio > 1/3) ###"
$PY -u train_cnn_camels.py --labels $L --label-key tsc_mr033 \
    --tag mr033 --out-dir $OUT --epochs 80

echo
echo "### mass ratio > 0.20 ###"
$PY -u train_cnn_camels.py --labels $L --label-key tsc_mr020 \
    --tag mr020 --out-dir $OUT --epochs 80

echo
echo "### mass ratio > 0.10 ###"
$PY -u train_cnn_camels.py --labels $L --label-key tsc_mr010 \
    --tag mr010 --out-dir $OUT --epochs 80
