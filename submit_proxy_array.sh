#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J proxy_snap
#SBATCH -a 71-99
#SBATCH -o logs/proxy_snap_%A_%a.out

cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u validate_merger_proxy.py --stage snapshot \
    --snap ${SLURM_ARRAY_TASK_ID}
