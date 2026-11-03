#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=spatial-pilot
#SBATCH --exclude=gpu3001,gpu2504,gpu3103
#SBATCH --output=logs/spatial_pilot_%j.out

# One job: 3 arms x 5 seeds sequentially; no array or background watcher.
# Pass the prepared run directory, containing a frozen source snapshot.
set -euo pipefail
ROOT=/oscar/data/idellant/cluster-ml
RUN_DIR=${1:?pass the prepared run directory}
WORK_DIR=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cluster-spatial.${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$WORK_DIR/matplotlib"
cd "$ROOT"
"$ROOT/venv/bin/python" -u "$RUN_DIR/code/test_spatial_pilot.py"
"$ROOT/venv/bin/python" -u "$RUN_DIR/code/spatial_pilot.py" run \
    --root "$ROOT" --out "$RUN_DIR" --work "$WORK_DIR/data" \
    --seeds 42,43,44,45,46 --epochs 60 --batch-size 32
