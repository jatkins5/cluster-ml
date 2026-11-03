#!/bin/bash
#SBATCH --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00
#SBATCH --job-name=syn-offset
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
RUN=${1:?pass the follow-up run directory}
shift
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
./venv/bin/python -u "$RUN/code/analyze_radio_xray_offset.py" --run-dir "$RUN" "$@"
