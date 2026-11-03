#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=12:00:00
#SBATCH --job-name=syn-xray-control
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
# Five seeds run serially in ONE GPU allocation, matching the active syntrain protocol.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
RUN=${1:?pass the follow-up run directory}
test -s "$RUN/preflight/verified.json"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
for SEED in 42 43 44 45 46; do
    PREFIX="$RUN/preds/xray_b3000_s$SEED"
    if test -e "${PREFIX}_preds.npz"; then
        echo "Refusing to overwrite ${PREFIX}_preds.npz" >&2
        exit 1
    fi
    ./venv/bin/python -u "$RUN/code/train_cnn_mock.py" \
        --dataset xray_fix_b3000_s9991.h5 --seed "$SEED" --out-prefix "$PREFIX"
done
