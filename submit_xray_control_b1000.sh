#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=12:00:00
#SBATCH --job-name=xray-ctrl-1mpc
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH -o logs/xray_ctrl_b1000_%j.out
# Matched X-ray-only control for the 1 Mpc radio + X-ray joint model
# (syn_phys_b1000_joint): same X-ray inputs as that joint (xray_fix_s9991.h5),
# same default outer folds and pooled-view inner selection, same frozen
# defining modules (runs/followup_20261103T180830Z/code, hash-identical to the
# versions recorded in the joint archives). Mirrors submit_syn_xray_control.sh
# with only the dataset changed. The preflight (review round 59) runs first in
# this allocation and stops the job before any training on any mismatch.
# Five seeds serially in one GPU allocation.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
RUN=runs/xray_control_b1000
CODE=runs/followup_20261103T180830Z/code/train_cnn_mock.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
./venv/bin/python -u preflight_xray_control_b1000.py --run-dir "$RUN"
grep -q '"status": "passed"' "$RUN/preflight/verified.json"
mkdir -p "$RUN/preds"
for SEED in 42 43 44 45 46; do
    PREFIX="$RUN/preds/xray_b1000_s$SEED"
    if test -e "${PREFIX}_preds.npz"; then
        echo "Refusing to overwrite ${PREFIX}_preds.npz" >&2
        exit 1
    fi
    ./venv/bin/python -u "$CODE" --dataset xray_fix_s9991.h5 --seed "$SEED" --out-prefix "$PREFIX"
done
