#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=16G -t 00:20:00
#SBATCH -J chkxmock
#SBATCH -o logs/check_xray_mock_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u check_xray_mock.py
