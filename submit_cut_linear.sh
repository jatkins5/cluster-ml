#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=24G -t 03:00:00
#SBATCH -J cutlin
#SBATCH -o logs/cut_linear_%j.out

cd /oscar/data/idellant/cluster-ml

# The compressed map barely notices the density cut, which is expected:
# arcsinh(w) ~ log(w) already flattens the spike. The cut should matter on the
# LINEAR map, where one cell previously took ~22% of the box flux. Without a
# cut, linear scored 0.620 / 0.350 / 0.375 with R2 +0.286.
for TAG in nh3 nh4; do
  [ -f "dataset_${TAG}_512.h5" ] || continue
  echo "################ linear (inverted), n_H cut $TAG ################"
  ./venv/bin/python -u forward_model_lotss.py \
      --dataset "dataset_${TAG}_512.h5" \
      --mask-compact --mask-max-beams 10 --invert-arcsinh \
      --out-prefix "forward_${TAG}_lin"
done
