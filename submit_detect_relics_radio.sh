#!/bin/bash
#SBATCH --job-name=cluster-ml-relics-radio
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/relics_radio_%j.out
#SBATCH --error=logs/relics_radio_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# Run radio-map peak detection on the existing dataset_512 (no cutout I/O).
# Then re-use the v3 validation script to produce an analogous figure for
# direct comparison with the Mach-detector result.
python detect_relics_radio.py --output relic_catalog_radio.h5

python plot_relic_validation_v3.py \
    --catalog relic_catalog_radio.h5 \
    --tsc-lo 0.3 --tsc-hi 3.0 --mr-thresh 0.1 \
    --out relic_radio_validation.png
