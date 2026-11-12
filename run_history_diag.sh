#!/bin/bash
#SBATCH -p batch -n 1 -c 4 --mem=16G -t 02:00:00
#SBATCH -J histdiag
#SBATCH -o logs/history_diag_%j.out
# Phase 5 (B1/B2 merger-history diagnostics) on saved labels/predictions; CPU only, no training.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u diagnose_history.py \
    --expect-manifest f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d \
    --out history_diag_${SLURM_JOB_ID:?run under SLURM}.npz
