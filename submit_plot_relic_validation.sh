#!/bin/bash
#SBATCH --job-name=cluster-ml-relicval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/relicval_%j.out
#SBATCH --error=logs/relicval_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate
python plot_relic_validation.py
