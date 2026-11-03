#!/bin/bash
#SBATCH --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:15:00
#SBATCH --job-name=syn-pair-check
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
RUN=${1:?pass the follow-up run directory}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
./venv/bin/python -u "$RUN/code/check_syn_pairing.py" --run-dir "$RUN"
