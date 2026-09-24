#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 2 --mem=8G -t 00:30:00
#SBATCH -J soxs
#SBATCH -o logs/install_soxs_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/pip install soxs 2>&1 | tail -3
./venv/bin/python - <<'PY'
import soxs
print("soxs", soxs.__version__)
from soxs.instrument_registry import instrument_registry
print("chandra_acisi_cy22" in instrument_registry.keys())
print(instrument_registry["chandra_acisi_cy22"])
PY
