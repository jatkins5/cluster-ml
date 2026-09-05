#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J mask_sweep
#SBATCH -o logs/mask_sweep_%j.out

cd /oscar/data/idellant/cluster-ml
PY=./venv/bin/python

echo "### control: no masking ###"
$PY -u forward_model_lotss.py --out-prefix forward_lotss_ctl
echo

for MB in 2 5 10 20; do
  echo "### obs-only masking, max_beams=$MB ###"
  $PY -u forward_model_lotss.py --mask-compact --mask-max-beams $MB \
      --out-prefix forward_lotss_obsmask${MB}
  echo
done
