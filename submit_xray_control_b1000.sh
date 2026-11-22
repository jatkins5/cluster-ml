#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=12:00:00
#SBATCH --job-name=xray-ctrl-1mpc
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/xray_ctrl_b1000_%j.out
# Matched X-ray-only control for the 1 Mpc radio + X-ray joint model
# (syn_phys_b1000_joint): same X-ray inputs as that joint (xray_fix_s9991.h5),
# same default outer folds and pooled-view inner selection, same frozen trainer
# (runs/followup_20261103T180830Z/code/train_cnn_mock.py, identical to the
# version the joint runs used). Mirrors submit_syn_xray_control.sh with only
# the dataset changed. Five seeds serially in one GPU allocation.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
CODE=runs/followup_20261103T180830Z/code/train_cnn_mock.py
OUT=runs/xray_control_b1000/preds
mkdir -p "$OUT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
sha256sum "$CODE" xray_fix_s9991.h5
for SEED in 42 43 44 45 46; do
    PREFIX="$OUT/xray_b1000_s$SEED"
    if test -e "${PREFIX}_preds.npz"; then
        echo "Refusing to overwrite ${PREFIX}_preds.npz" >&2
        exit 1
    fi
    ./venv/bin/python -u "$CODE" --dataset xray_fix_s9991.h5 --seed "$SEED" --out-prefix "$PREFIX"
done
