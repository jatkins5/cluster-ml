#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J cutsyn
#SBATCH -o logs/cut_syn_%j.out
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u - <<'PY'
from compare_f7 import r2s, paired
paired("OLD model: no cut -> n_H < 1e-4", r2s("pooled_nocut144_inner_s{}.npz"), r2s("pooled_nh4_144_inner_s{}.npz"))
paired("NEW model: no cut -> n_H < 1e-4", r2s("pooled_nocut_syn_inner_s{}.npz"), r2s("pooled_syn_inner_s{}.npz"))
paired("cut maps: old -> new model", r2s("pooled_nh4_144_inner_s{}.npz"), r2s("pooled_syn_inner_s{}.npz"))
paired("no-cut maps: old -> new model", r2s("pooled_nocut144_inner_s{}.npz"), r2s("pooled_nocut_syn_inner_s{}.npz"))
PY
