#!/bin/bash
#SBATCH --job-name=build-xray
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/build_xray_%j.out
#SBATCH --error=logs/build_xray_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

echo "=== Building 128px X-ray dataset ==="
python build_xray_dataset.py --img-size 128 --output dataset_xray_128.h5

echo ""
echo "=== Building 256px X-ray dataset ==="
python build_xray_dataset.py --img-size 256 --output dataset_xray_256.h5
