#!/bin/bash
#SBATCH -p batch -n 1 -c 4 --mem=8G -t 00:40:00
#SBATCH -J histfix
#SBATCH -o logs/history_fixtures_%j.out
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export TMPDIR=$(mktemp -d -p /tmp histfix.XXXX)
./venv/bin/python -u test_history.py
rm -rf "$TMPDIR"
