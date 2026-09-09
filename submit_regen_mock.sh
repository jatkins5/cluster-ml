#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 04:00:00
#SBATCH -J regenmock
#SBATCH -o logs/regen_mock_%j.out
cd /oscar/data/idellant/cluster-ml
set -e
for T in gp nh3 nh4; do
  ./venv/bin/python -u build_mock_dataset.py --dataset "dataset_${T}_512.h5" \
      --output "mockcn_${T}.h5"
  echo "############ domain gap, $T ############"
  ./venv/bin/python -u check_domain_gap.py --dataset "mockcn_${T}.h5"
done
