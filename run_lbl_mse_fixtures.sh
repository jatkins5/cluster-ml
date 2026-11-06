#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J lblmsefx
#SBATCH -o logs/lbl_mse_fixtures_%j.out
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export TMPDIR=$(mktemp -d -p /tmp lblmse.XXXX)
./venv/bin/python -u test_lbl_mse.py
rm -rf "$TMPDIR"
