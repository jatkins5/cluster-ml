#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 04:00:00
#SBATCH -J phasesyn
#SBATCH -o logs/phase_syn_%j.out
# Stale regeneration, job A (W3): per-cell gas state and 144/1400 MHz ratios of
# the current synch weights from the raw snapshot-99 cutouts (read-only).
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
./venv/bin/python -u extract_phase_syn.py --out phase_cells_syn_144_${SLURM_JOB_ID:?run under SLURM}.npz
