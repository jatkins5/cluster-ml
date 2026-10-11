#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J e5unit
#SBATCH -o logs/smoke_e5_unit_%j.out
# E5 fixture checks without training (review round 16).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export TMPDIR=$(mktemp -d -p /tmp/ e5unit.XXXX)
./venv/bin/python -u test_e5_fixtures.py
rm -rf "$TMPDIR"
