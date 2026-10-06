"""Are A399 / A401 still the youngest real clusters, and is that mass-driven?

For each model's seed ensemble: their predicted ages and ranks (1 = youngest)
among the real targets, and the rank correlation of all predictions with the
weak-lensing mass. The mass-only model (same training set, images zeroed) is
the reference: if it also puts them first, their youth comes from their mass.
"""
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
key = lambda s: "".join(str(s).split()).replace("_", "").upper()
WATCH = ["A399", "A401"]

MODELS = [
    ("99, mass only", "b2_sfmassonly_s*_preds.npz"),
    ("99, radio", "nu144_sfimg_s*_preds.npz"),
    ("99, radio + mass", "nu144_sfimgmassn_s*_preds.npz"),
    ("99, X-ray", "x144_s*_preds.npz"),
    ("99, joint", "joint144_s*_preds.npz"),
    ("99+91, mass only", "s9991_massonly_s*_preds.npz"),
    ("99+91, radio", "s9991_sfimg_s*_preds.npz"),
    ("99+91, radio + mass", "s9991_sfimgmassn_s*_preds.npz"),
    ("99+91, X-ray", "s9991_x_s*_preds.npz"),
    ("99+91, joint", "s9991_joint_s*_preds.npz"),
]

print(f"{'model':<22}{'seeds':>6}{'n':>4}  " + "".join(f"{w:>16}" for w in WATCH)
      + f"{'rho(pred, M_WL)':>18}")
for name, pat in MODELS:
    files = sorted(glob.glob(f"cnn_preds/{pat}"))
    if not files:
        print(f"{name:<22}  (no runs)")
        continue
    ds = [np.load(p) for p in files]
    names = [key(n) for n in ds[0]["obs_name"]]
    pred = np.mean([d["obs_pred"] for d in ds], axis=0)
    rank = stats.rankdata(pred)
    cells = ""
    for w in WATCH:
        if w in names:
            i = names.index(w)
            cells += f"{pred[i]:>8.2f} ({int(rank[i]):>2}/{len(pred)})"
        else:
            cells += f"{'-':>16}"
    m = [n in wl.index for n in names]
    rho = stats.spearmanr(pred[m], [wl.loc[n, "log_m500c"] for n, k in zip(names, m) if k])
    print(f"{name:<22}{len(files):>6}{len(pred):>4}  {cells}{rho[0]:>+10.2f} (p={rho[1]:.2f})")

lm = wl.loc[[w for w in WATCH if w in wl.index], "log_m500c"]
allm = wl.loc[wl.has_lotss.astype(bool), "log_m500c"].rank(ascending=False)
print("\nweak-lensing masses: " + ", ".join(
    f"{k} log M = {v:.2f} (rank {int(allm[k])} heaviest of {len(allm)} LOFAR targets)"
    for k, v in lm.items()))
