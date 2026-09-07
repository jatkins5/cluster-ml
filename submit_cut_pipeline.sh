#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=64G -t 08:00:00
#SBATCH -J cutpipe
#SBATCH -o logs/cut_pipeline_%j.out

cd /oscar/data/idellant/cluster-ml
set -e

# n_H < 1e-3 keeps 96.8% of shock cells while dropping the brightest cell's
# share from 21.9% to 9.3%; 1e-4 concentrates less still but discards a
# quarter of the cells. Test both. The absolute weight lost is irrelevant --
# the forward model renormalises total flux to the Cuciti anchor, so only the
# spatial distribution matters.
for TAG in nh3 nh4; do
  case $TAG in
    nh3) NH=1e-3 ;;
    nh4) NH=1e-4 ;;
  esac
  echo "################ regenerating cells, n_H < $NH ################"
  ./venv/bin/python -u build_radio_cells.py --max-nh "$NH" \
      --out-dir "Radio_Cells_$TAG"

  echo "################ building 512px dataset, n_H < $NH ################"
  ./venv/bin/python -u build_dataset.py --center grouppos \
      --cells-dir "Radio_Cells_$TAG" --img-size 512 \
      --output "dataset_${TAG}_512.h5"

  echo "################ forward model, n_H < $NH ################"
  ./venv/bin/python -u forward_model_lotss.py \
      --dataset "dataset_${TAG}_512.h5" \
      --mask-compact --mask-max-beams 10 \
      --out-prefix "forward_${TAG}"
done
