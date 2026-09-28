#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=32G -t 00:30:00
#SBATCH -J radval
#SBATCH -o logs/radio_validate_%j.out
# After adding --z / --nu-ghz: the defaults must still reproduce the stored
# upstream weights exactly.
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u build_radio_cells.py --validate --limit 5 --out-dir /tmp/radval_$SLURM_JOB_ID
