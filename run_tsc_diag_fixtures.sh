#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:20:00
#SBATCH -J tscfix
#SBATCH -o logs/tsc_diag_fixtures_%j.out
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export TMPDIR=$(mktemp -d -p /tmp tscfix.XXXX)
./venv/bin/python -u test_tsc_diag.py
rm -rf "$TMPDIR"
