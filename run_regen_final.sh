#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 8 --mem=32G -t 03:00:00
#SBATCH -J regenFin
#SBATCH -o logs/regen_final_%j.out
# Stale regeneration, consolidated final run: all nine workstreams at one
# committed revision, consuming the validated job A (8941185) and job B
# (8941186) products, so a single promotable manifest covers every item.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
./venv/bin/python -u paper/scripts/regen_stale.py \
    --run-dir paper/staged/regen_final_${SLURM_JOB_ID:?run under SLURM} \
    --tasks w1,w2,w3,w4,w5,w6,w7,w8,w9 --phase-npz phase_cells_syn_144_8941185.npz
