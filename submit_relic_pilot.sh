#!/bin/bash
#SBATCH --partition=batch --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=04:00:00
#SBATCH --job-name=relic-pilot
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
RUN=${1:?pass the follow-up run directory}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
./venv/bin/python -u "$RUN/code/pilot_relic_association.py" --run-dir "$RUN"
