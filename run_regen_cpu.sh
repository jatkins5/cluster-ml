#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 8 --mem=32G -t 03:00:00
#SBATCH -J regencpu
#SBATCH -o logs/regen_cpu_%j.out
# Stale regeneration, job C: W1, W4-W9 on existing current inputs (CPU; ridge
# and statistics only, no CNN refits). Staged outputs only.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
./venv/bin/python -u paper/scripts/regen_stale.py \
    --run-dir paper/staged/regen_C_${SLURM_JOB_ID:?run under SLURM} --tasks w1,w4,w5,w6,w7,w8,w9
