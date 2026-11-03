#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 03:00:00
#SBATCH -J valsynch
#SBATCH -o logs/validate_synch_%j.out
# Port check: build_radio_cells.py --model synch vs Chuiyang's Radio_Snap{99,91}_new
# weights (no cuts, 144 MHz, w-scale 1), 20 halos per snapshot; and the
# legacy hb07 path vs the original Radio_Data weights (3 halos).
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
C=/oscar/data/idellant/Chuiyang
T=$(mktemp -d -p /tmp valsynch.XXXX)
PY="./venv/bin/python -u"
echo "== snapshot 99 (z = 0)"
$PY build_radio_cells.py --model synch --validate --nu-ghz 0.144 --limit 20 \
    --radio-dir $C/Radio_Snap99_new --cutout-dir $C/TNGCluster_Cutout/snap99 --out-dir $T/s99
echo "== snapshot 91 (z = 0.0994)"
$PY build_radio_cells.py --model synch --validate --nu-ghz 0.144 --limit 20 \
    --z 0.0994018026302219 \
    --radio-dir $C/Radio_Snap91_new --cutout-dir $C/TNGCluster_Cutout/snap91 --out-dir $T/s91
echo "== legacy hb07 regression"
$PY build_radio_cells.py --validate --limit 3 --out-dir $T/hb07
rm -rf "$T"
