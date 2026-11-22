#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:30:00
#SBATCH -J cleanJbs
#SBATCH -o logs/clean_joint_bootstrap_%j.out
# Paired parent bootstrap for the clean radio -> radio + X-ray increment (no training).
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
PYTHONPATH=. ./venv/bin/python -u score_clean_joint_bootstrap.py \
    --out runs/clean_joint_bootstrap_${SLURM_JOB_ID:?run under SLURM}.json
