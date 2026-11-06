#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=16G -t 01:00:00
#SBATCH -J tscdiag
#SBATCH -o logs/tsc_diag_%j.out
# Phase 4 Tier 0 (A0-A3) on saved labels/predictions; CPU only, no training.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u diagnose_tsc_hardness.py \
    --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d \
    --out tsc_hardness_diag_${SLURM_JOB_ID:?run under SLURM}.npz
