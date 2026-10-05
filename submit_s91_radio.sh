#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 12:00:00
#SBATCH -J s91radio
#SBATCH -o logs/s91_radio_%j.out
# Snapshot 91 (z = 0.0994), radio side, with the snapshot-99 configuration:
# 144 MHz, n_H < 1e-4 shock cells in 1.4 GHz units (--w-scale 12.5),
# GroupPos centring, +-4 r500 at that snapshot's own R500, and mocks
# injected into the same LoTSS fields with the z=0 flux constant.
cd /oscar/data/idellant/cluster-ml
set -e
C=/oscar/data/idellant/Chuiyang
Z91=0.0994018026302219

./venv/bin/python -u build_radio_cells.py --max-nh 1e-4 --nu-ghz 0.144 \
    --w-scale 12.5 --z $Z91 \
    --radio-dir $C/Radio_Snap91 --cutout-dir $C/TNGCluster_Cutout/snap91 \
    --out-dir Radio_Cells_nh4_144_s91

for S in 512 128; do
  ./venv/bin/python -u build_dataset.py --snap 91 --center grouppos \
      --groupcat $C/groupcat_classification/groupcat_091 \
      --halo-map $C/TNGCluster_Cutout/fof_halo_to_sub91.csv \
      --cells-dir Radio_Cells_nh4_144_s91 --img-size $S \
      --output dataset_nh4_144_s91_${S}.h5
done

OFF=$(./venv/bin/python -c "import h5py; print(h5py.File('injected_nh4_144_simflux.h5').attrs['sim_flux_offset_dex'])")
echo "z=0 sim flux offset: $OFF"
./venv/bin/python -u build_injected_dataset.py \
    --dataset dataset_nh4_144_s91_512.h5 --field-dir lotss_fields_40 \
    --flux-mode sim --sim-offset "$OFF" --seed 91 \
    --output injected_nh4_144_s91_simflux.h5
