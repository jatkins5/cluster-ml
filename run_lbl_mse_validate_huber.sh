#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=4G -t 00:10:00
#SBATCH -J lblmseval
#SBATCH -o logs/lbl_mse_validate_huber_%j.out
# Validators only: the existing Huber catalogue-TSC artifacts against the
# comparator's per-artifact contract (no scoring, no MSE arm needed).
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u - <<'PY'
import json, numpy as np
import compare_labels as L, compare_lbl_mse as M
for s in L.PRODUCTION_SEEDS:
    d = dict(np.load(f"cnn_preds/lbl_x_tsc_tcoll_s{s}_preds.npz", allow_pickle=True))
    L.validate_artifact(f"huber seed {s}", d, 60)
    M.validate_semantics(f"huber seed {s}", d, json.loads(str(d["provenance"])), 60)
    print(f"  huber seed {s}: per-artifact and absolute checks passed (loss {d['loss']})")
print("EXISTING HUBER ARTIFACTS PASS THE COMPARATOR CONTRACT")
PY
