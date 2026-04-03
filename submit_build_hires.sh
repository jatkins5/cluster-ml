#!/bin/bash
#SBATCH --job-name=build-hires
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/build_hires_%j.out
#SBATCH --error=logs/build_hires_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== Building 256px, 4×R500c ==="
python build_dataset.py --img-size 256 --output dataset_256.h5

echo ""
echo "=== Building 256px, 2×R500c ==="
python build_dataset.py --img-size 256 --extent-r500 2.0 --output dataset_256_2r500.h5
