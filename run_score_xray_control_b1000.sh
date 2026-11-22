#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 01:00:00
#SBATCH -J scoreXc1
#SBATCH -o logs/score_xray_ctrl_b1000_%j.out
# Frozen matched scoring for the 1 Mpc X-ray control (after submit_xray_control_b1000.sh).
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
PYTHONPATH=. ./venv/bin/python -u score_xray_control_b1000.py \
    --out runs/xray_control_b1000/score_${SLURM_JOB_ID:?run under SLURM}.json
