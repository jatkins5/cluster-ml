#!/bin/bash
#SBATCH -p batch -n 1 -c 2 --mem=8G -t 00:20:00
#SBATCH -J lblunit
#SBATCH -o logs/smoke_labels_unit_%j.out
# Catalogue-TSC label fixtures without training.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export TMPDIR=$(mktemp -d -p /tmp lblunit.XXXX)
./venv/bin/python -u test_label_fixtures.py
rm -rf "$TMPDIR"
