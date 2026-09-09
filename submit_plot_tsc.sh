#!/bin/bash
#SBATCH --job-name=plot-tsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/plot_tsc_%j.out
#SBATCH --error=logs/plot_tsc_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

python plot_tsc_predictions.py
