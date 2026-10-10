#!/bin/bash
#SBATCH -p batch -n 1 -c 1 --mem=8G -t 00:15:00
#SBATCH -J smoker10
#SBATCH -o logs/smoke_r10_%j.out
# Review round 10: compare_e1x.py must reject a required common control that
# is missing CONSISTENTLY from every run (the round-9 tests only mutated one
# run). Uses the saved round-9 fixture; no training.
set -euo pipefail
cd /oscar/data/idellant/cluster-ml
./venv/bin/python -u - <<'PY'
import copy, json
import numpy as np
import compare_e1x as C
MAN = "f08136fd6d1a32083c8ea6287fa6a64ff350e8bf2ca5e0a2a7187cf33461df4d"
runs = C.load("smoke_r9_tmp", [42, 43])
C.check(runs, MAN)
print("  valid fixture still accepted")
def cfg_set(k, v):
    def f(d):
        p = json.loads(str(d["provenance"])); p["config"][k] = v
        d["provenance"] = np.array(json.dumps(p))
    return f
def drop_input(name):
    def f(d):
        p = json.loads(str(d["provenance"])); p["inputs_sha256"].pop(name)
        d["provenance"] = np.array(json.dumps(p))
    return f
def no_noise(d):
    cfg_set("mass_noise", False)(d); d["global_cols"] = np.array(["log_m500", "z_placement"])
def no_z(d):
    cfg_set("redshift_scalar", False)(d)
    d["global_cols"] = np.array(["log_m500_noisy"]); d["n_scalar"] = np.array(1)
    d["global_scalar_raw"] = d["global_scalar_raw"][:, :1]
    d["fold_scalar_mu"] = d["fold_scalar_mu"][:, :1]; d["fold_scalar_sd"] = d["fold_scalar_sd"][:, :1]
def no_cols_only(d):           # metadata dropped but config flags intact
    d["global_cols"] = np.array(["log_m500_noisy"]); d["n_scalar"] = np.array(1)
cases = {"mass noise off in every arm": no_noise,
         "redshift omitted in every arm": no_z,
         "global columns missing z in every arm": no_cols_only,
         "final-epoch selection in every arm": cfg_set("select", "final"),
         "different mass seed in every arm": cfg_set("mass_seed", 1),
         "mass off in every arm": cfg_set("mass", False),
         "weak-lensing-mass hash absent in every run": drop_input("wl_masses"),
         "catalogue hash absent in every run": drop_input("catalog")}
for name, f in cases.items():
    r = copy.deepcopy(runs)
    for a in r:
        for s in r[a]:
            f(r[a][s])
    try:
        C.check(r, MAN)
    except SystemExit as e:
        print(f"  rejected ({name}): {e}")
        continue
    raise SystemExit(f"NOT rejected: {name}")
print("ALL ROUND-10 CHECKS PASSED")
PY
