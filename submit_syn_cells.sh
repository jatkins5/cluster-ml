#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 16:00:00
#SBATCH -J syncells
#SBATCH --array=0-1
#SBATCH -o logs/syn_cells_%A_%a.out
# Chuiyang's regenerated radio model (Radio_Snap{99,91}_new, 2026-11-02):
# --model synch, 144 MHz, our n_H < 1e-4 shock-cell cut, GroupPos centring,
# +-4 r500 at each snapshot's own R500 -- the nh4_144 configuration with only
# the weight model changed. --w-scale 206 = 12.5 x 16.5 (the old 1.4 GHz-unit
# scale times the sample-median new/old total-power ratio) keeps the arcsinh
# maps in the old units, so the comparison isolates the per-cell change.
#   task 0: snapshot 99 (+ shape-only maps and the relic catalogue)
#   task 1: snapshot 91 (z = 0.0994)
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
C=/oscar/data/idellant/Chuiyang
PY="./venv/bin/python -u"
WS=206
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  $PY build_radio_cells.py --model synch --max-nh 1e-4 --nu-ghz 0.144 --w-scale $WS \
      --radio-dir $C/Radio_Snap99_new --cutout-dir $C/TNGCluster_Cutout/snap99 \
      --out-dir Radio_Cells_syn_nh4_144
  for S in 512 128; do
    $PY build_dataset.py --center grouppos --cells-dir Radio_Cells_syn_nh4_144 \
        --img-size $S --output dataset_syn_nh4_144_${S}.h5
  done
  $PY build_shape_dataset.py --input dataset_syn_nh4_144_128.h5 \
      --output dataset_syn_nh4_144_128_shape.h5
  $PY detect_relics_radio.py --dataset dataset_syn_nh4_144_512.h5 \
      --output relic_catalog_radio_syn_nh4_144.h5
else
  Z91=0.0994018026302219
  $PY build_radio_cells.py --model synch --max-nh 1e-4 --nu-ghz 0.144 --w-scale $WS --z $Z91 \
      --radio-dir $C/Radio_Snap91_new --cutout-dir $C/TNGCluster_Cutout/snap91 \
      --out-dir Radio_Cells_syn_nh4_144_s91
  for S in 512 128; do
    $PY build_dataset.py --snap 91 --center grouppos \
        --groupcat $C/groupcat_classification/groupcat_091 \
        --halo-map $C/TNGCluster_Cutout/fof_halo_to_sub91.csv \
        --cells-dir Radio_Cells_syn_nh4_144_s91 --img-size $S \
        --output dataset_syn_nh4_144_s91_${S}.h5
  done
fi
