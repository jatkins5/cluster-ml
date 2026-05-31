#!/bin/bash
#SBATCH --job-name=cluster-ml-cond-nn
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cond_nn_%j.out
#SBATCH --error=logs/cond_nn_%j.err

mkdir -p logs
cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

python plot_cond_nn_check.py
