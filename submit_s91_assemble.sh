#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=96G -t 06:00:00
#SBATCH -J s91asm
#SBATCH -o logs/s91_assemble_%j.out
# After submit_s91_radio.sh and submit_s91_xray.sh:
#   1. join the snapshot-91 X-ray count parts and attach their labels
#   2. place both snapshots' X-ray mocks at their radio mocks' redshifts and
#      real exposures (snapshot 99 rebuilt on the 144 MHz radio set; 91 on
#      the same stretch constant so the two can be combined)
#   3. merge snapshots 99 + 91 for the clean, injected-radio and X-ray sets
#   4. attach the real Chandra inputs to the X-ray sets
cd /oscar/data/idellant/cluster-ml
set -e
PY=./venv/bin/python

$PY -u xray_counts_regrid.py --merge \
    --labels-from dataset_nh4_144_s91_128.h5 --output xray_counts_2Ms_128_s91.h5

$PY -u build_xray_realistic.py build --z-from injected_nh4_144_simflux.h5 \
    --lx-scale 0.2778 --nh-csv lofar_nh.csv --output xray_placed_144.h5
A=$($PY -c "import h5py; print(h5py.File('xray_placed_144.h5').attrs['stretch_scale'])")
echo "snapshot-99 X-ray stretch constant: $A"
$PY -u build_xray_realistic.py build --counts-h5 xray_counts_2Ms_128_s91.h5 \
    --z-from injected_nh4_144_s91_simflux.h5 --lx-scale 0.2778 \
    --nh-csv lofar_nh.csv --stretch-scale "$A" --seed 91 \
    --output xray_placed_144_s91.h5

$PY -u merge_snapshots.py dataset_nh4_144_128.h5 dataset_nh4_144_s91_128.h5 \
    --output dataset_nh4_144_s9991_128.h5
$PY -u merge_snapshots.py injected_nh4_144_simflux.h5 injected_nh4_144_s91_simflux.h5 \
    --output injected_nh4_144_s9991_simflux.h5
$PY -u merge_snapshots.py xray_placed_144.h5 xray_placed_144_s91.h5 \
    --output xray_placed_144_s9991.h5

for X in xray_placed_144.h5 xray_placed_144_s9991.h5; do
  $PY -u attach_xray_obs.py --obs-h5 xray_obs_acisi_pbc.h5 --train-h5 $X
done
$PY -u check_s91.py
