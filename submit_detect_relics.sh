#!/bin/bash
#SBATCH --job-name=cluster-ml-relics
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/relics_%j.out
#SBATCH --error=logs/relics_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Fail-fast smoke on halo 0 (~5 GB read + a few seconds of numpy). If this
# errors we'll see it within ~1 min and abort before the full 352-halo loop.
echo "=== smoke test: halo 0 ==="
python detect_relics.py --halo-id 0 --output relic_smoke.h5 || exit 1

echo "=== full run: all 352 halos ==="
python detect_relics.py --output relic_catalog.h5
