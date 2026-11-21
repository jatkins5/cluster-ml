#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=16G -t 01:00:00
#SBATCH -J regenD
#SBATCH -o logs/regen_after_%j.out
# Stale regeneration, job D: W2 (144 MHz and paired 1.4 GHz Minkowski XGBoost
# fits) and W3 (phase/frequency), after jobs A and B.
# Usage: sbatch --export=ALL,PHASE=phase_cells_syn_144_<jobA>.npz run_regen_after.sh
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
: "${PHASE:?pass PHASE=<job A npz>}"
[ -f "$PHASE" ] && [ -f dataset_syn_nh4_1400_128.h5 ] || { echo "job A/B outputs missing"; exit 1; }
./venv/bin/python -u paper/scripts/regen_stale.py \
    --run-dir paper/staged/regen_D_${SLURM_JOB_ID:?run under SLURM} --tasks w2,w3 --phase-npz "$PHASE"
