#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 01:00:00
#SBATCH -J xrbuild
#SBATCH -a 0-4
#SBATCH -o logs/xray_real_build_%A_%a.out
cd /oscar/data/idellant/cluster-ml
DEPTHS=(10 30 100 archive 2000)
D=${DEPTHS[$SLURM_ARRAY_TASK_ID]}
./venv/bin/python -u build_xray_realistic.py build --depth "$D" \
    --output "xray_real_${D}.h5"
