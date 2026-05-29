#!/bin/bash
#SBATCH --job-name=cluster-ml-relics-v2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/relics_v2_%j.out
#SBATCH --error=logs/relics_v2_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# v2 detector: stricter shock isolation to target merger bow shocks rather
# than the soup of internal/accretion shocks the v1 (M>=2, 5%, 300kpc) caught.
#   Mach>=3       :  ~half as many candidate cells (merger-shock-typical)
#   peak>=25%     :  5x stricter, drops faint sub-peaks
#   sep>=600 kpc  :  closer to observed double-relic spacing
#   r_max=2.0R500 :  cuts far-field accretion shocks
# Output to relic_catalog_v2.h5; validation uses the pre-joined merger-TSC
# (TSC_eachhalo_snap99.hdf5) and stratifies by mass_ratio.

echo "=== smoke: halo 0 (v2 params) ==="
python detect_relics.py --halo-id 0 --output relic_smoke_v2.h5 \
    --mach-thresh 3.0 --peak-thresh-frac 0.25 --min-sep-kpc 600 \
    --r-max-frac 2.0 || exit 1

echo "=== full v2 detection (all 352) ==="
python detect_relics.py --output relic_catalog_v2.h5 \
    --mach-thresh 3.0 --peak-thresh-frac 0.25 --min-sep-kpc 600 \
    --r-max-frac 2.0

echo "=== validation plot (merger-TSC, mass-ratio stratified) ==="
python plot_relic_validation.py \
    --catalog relic_catalog_v2.h5 \
    --outprefix relic_v2
