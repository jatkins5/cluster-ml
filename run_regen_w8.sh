#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:30:00
#SBATCH -J regenW8
#SBATCH -o logs/regen_w8_%j.out
# Stale regeneration: rerun of W8 alone after the rounded-value reproduction
# check fix (job C 8941187 failed W8 on rho = 0.675 vs reported 0.67).
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
./venv/bin/python -u paper/scripts/regen_stale.py \
    --run-dir paper/staged/regen_W8_${SLURM_JOB_ID:?run under SLURM} --tasks w8
