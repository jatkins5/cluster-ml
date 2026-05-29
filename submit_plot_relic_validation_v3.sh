#!/bin/bash
#SBATCH --job-name=cluster-ml-relicval3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/relicval3_%j.out
#SBATCH --error=logs/relicval3_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate
python plot_relic_validation_v3.py \
    --catalog relic_catalog_v2.h5 \
    --tsc-lo 0.3 --tsc-hi 3.0 --mr-thresh 0.1 \
    --out relic_v3_validation.png
