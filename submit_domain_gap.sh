#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 01:00:00
#SBATCH -J domgap
#SBATCH -o logs/domain_gap_%j.out
cd /oscar/data/idellant/cluster-ml
for D in mock_dataset_nh4.h5 mock_dataset_nocut.h5; do
  echo "############ $D ############"
  ./venv/bin/python -u check_domain_gap.py --dataset "$D"
done
