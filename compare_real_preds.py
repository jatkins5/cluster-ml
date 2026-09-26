"""Real-cluster predictions from radio, X-ray and joint models, side by side.

There is no ground truth for these clusters, so what can be checked is:
  - do the models agree with each other (independent observables, so
    agreement is not automatic);
  - are predictions in the training range and not collapsed;
  - how strongly each tracks halo mass (weak-lensing M500) and raw
    brightness -- the confound that made the radio-only transfer model
    unreliable.
"""
import glob
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
SETS = {"radio": "cnn_preds/b2_sfimg_s*_preds.npz",
        "xray": "cnn_preds/xlx_s*_preds.npz",
        "joint": "cnn_preds/jointlx_s*_preds.npz"}

preds, mock = {}, {}
for k, pat in SETS.items():
    files = sorted(glob.glob(pat))
    if not files:
        continue
    per = [np.load(p, allow_pickle=True) for p in files]
    names = [str(n) for n in per[0]["obs_name"]]
    preds[k] = pd.Series(np.mean([d["obs_pred"] for d in per], axis=0),
                         index=[key(n) for n in names])
    mock[k] = [r2_score(d["tsc"], d["oof_cluster"]) for d in per]
    print(f"{k:<6} {len(files)} seeds  mock OOF R2 {np.mean(mock[k]):.3f} "
          f"+- {np.std(mock[k], ddof=1):.3f}   real targets {len(names)}")

t = pd.DataFrame(preds)
m = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")["log_m500c"]
t["logM_WL"] = m.reindex(t.index)
with h5py.File("xray_obs_acisi.h5") as f:
    xk = [key(s.decode() if isinstance(s, bytes) else s) for s in f["obs/key"][:]]
    xi = f["obs/images"][:].astype(np.float64)
    a = float(f.attrs["stretch_scale"])
t["xray_rate"] = pd.Series((np.sinh(xi) * a).mean(axis=(1, 2)), index=xk)

print("\n======== real predictions (Gyr since last collision)")
print(t.sort_values("joint" if "joint" in t else "radio")
      .to_string(float_format=lambda v: f"{v:.2f}"))

print("\n======== agreement and confounds (Spearman over clusters with both)")
cols = [c for c in ["radio", "xray", "joint"] if c in t]
for i, a_ in enumerate(cols):
    for b_ in cols[i + 1:] + ["logM_WL", "xray_rate"]:
        d = t[[a_, b_]].dropna()
        if len(d) >= 5:
            r, p = stats.spearmanr(d[a_], d[b_])
            print(f"  {a_:<6} vs {b_:<10} rho {r:+.2f}  p={p:.3f}  n={len(d)}")
print("\n  training labels span 0.1-4.0 Gyr, median 0.92")
for c in cols:
    v = t[c].dropna()
    print(f"  {c:<6} real: median {v.median():.2f}  sd {v.std():.2f}  "
          f"range {v.min():.2f}-{v.max():.2f}")
