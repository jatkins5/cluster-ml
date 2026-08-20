#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J cam_proxy
#SBATCH -a 0-15
#SBATCH -o logs/cam_proxy_%A_%a.out

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
CHUNK=48
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
./venv/bin/python -u build_camels_proxy_tsc.py --stage zooms \
    --start ${START} --end ${END} --force
