#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=8G -t 00:30:00
#SBATCH -J regenfix
#SBATCH -o logs/regen_fixtures_%j.out
# Synthetic fixtures for paper/scripts/regen_stale.py and extract_phase_syn.py.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$(mktemp -d -p /tmp regenfix.XXXX)
./venv/bin/python -u test_regen_stale.py
rm -rf "$TMPDIR"
