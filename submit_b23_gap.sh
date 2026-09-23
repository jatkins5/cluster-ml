#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 4 --mem=48G -t 00:40:00
#SBATCH -J b23gap
#SBATCH -o logs/b23_gap_%j.out
cd /oscar/data/idellant/cluster-ml
echo "################ anchored flux (current default)"
./venv/bin/python -u check_domain_gap.py --dataset injected_nh4.h5
echo
echo "################ simulation's own power"
./venv/bin/python -u check_domain_gap.py --dataset injected_nh4_simflux.h5
