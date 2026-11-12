#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 04:00:00
#SBATCH -J cells1400
#SBATCH -o logs/cells_1400_%j.out
# Stale regeneration, job B (W2/W3): density-cut 1.4 GHz synch cells and the
# 128-px map set, mirroring submit_syn_cells.sh task 0 (GroupPos, n_H <= 1e-4,
# +-4 R500 default extent, point deposition -- no --spread-cells) with only
# --nu-ghz changed. --w-scale 16.48 = 206/12.5 is a storage convention only.
# The source record is captured BEFORE the builders and re-verified after them;
# the success manifest is written only if nothing changed.
# PREPARED; run only after review and the user's go-ahead.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
C=/oscar/data/idellant/Chuiyang
OUTC=Radio_Cells_syn_nh4_1400
OUTD=dataset_syn_nh4_1400_128.h5
PY="./venv/bin/python -u"
MAN="$PY manifest_cells_1400.py --dataset $OUTD --cells-dir $OUTC --radio-dir $C/Radio_Snap99_new \
     --cutout-dir $C/TNGCluster_Cutout/snap99 --ratio-table Radio_Data_new/radio_spectral_ratio_table.npz \
     --nu-ghz 1.4 --w-scale 16.48 --max-nh 1e-4 --img-size 128"
$MAN --stage begin                       # refuses if any output already exists
$PY build_radio_cells.py --model synch --max-nh 1e-4 --nu-ghz 1.4 --w-scale 16.48 \
    --radio-dir $C/Radio_Snap99_new --cutout-dir $C/TNGCluster_Cutout/snap99 \
    --ratio-table Radio_Data_new/radio_spectral_ratio_table.npz --out-dir $OUTC
$PY build_dataset.py --center grouppos --cells-dir $OUTC --img-size 128 --output $OUTD
$MAN --stage end
