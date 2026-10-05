#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=24G -t 08:00:00
#SBATCH -J s91evt
#SBATCH --array=0-15
#SBATCH -o logs/s91_evt_%A_%a.out
# Regrid the snapshot-91 X-ray images (3 projections x 351 halos, 0.19 GB
# each) onto the snapshot-99 physical grid; task 0 also checks the split
# reproduces the snapshot-99 recipe exactly on the native grid.
cd /oscar/data/idellant/cluster-ml
CHECK=""
[ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && CHECK="--check"
./venv/bin/python -u xray_counts_regrid.py --ntasks 16 $CHECK
