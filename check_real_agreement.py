"""Is the radio / X-ray agreement on real clusters just shared mass?

Both models' predictions correlate (weakly) with weak-lensing mass, and both
modalities get brighter with mass, so agreement could be a common mass
signal rather than two observables seeing the same merger state. Partial
correlation at fixed mass, and at fixed redshift, on the clusters that have
both predictions (and a mass, for the first)."""
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def ens(pat):
    per = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(pat))]
    return pd.Series(np.mean([d["obs_pred"] for d in per], axis=0),
                     index=[key(n) for n in per[0]["obs_name"]])


def partial(x, y, z):
    rk = stats.rankdata
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rk(x), rk(z)), resid(rk(y), rk(z)))


t = pd.DataFrame({"radio": ens("cnn_preds/b2_sfimg_s*_preds.npz"),
                  "xray": ens("cnn_preds/xlx_s*_preds.npz"),
                  "xray_v1": ens("cnn_preds/xlx_v1_s*_preds.npz")})
m = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")
t["logM"] = m["log_m500c"].reindex(t.index)
tl = pd.read_csv("LoVoCCS_target_list - lovoccs.csv")
zser = pd.Series(pd.to_numeric(tl["redshift"], errors="coerce").values,
                 index=tl["name"].map(key))
t["z"] = zser[~zser.index.duplicated()].reindex(t.index)   # target list repeats some names

for x in ["xray", "xray_v1"]:
    d = t[["radio", x, "z"]].dropna()
    r, p = stats.spearmanr(d.radio, d[x])
    rz, pz = partial(d.radio.values, d[x].values, d.z.values)
    print(f"radio vs {x:<8} n={len(d):2d}  rho {r:+.2f} (p={p:.3f})   "
          f"at fixed z {rz:+.2f} (p={pz:.3f})")
    dm = t[["radio", x, "logM"]].dropna()
    r, p = stats.spearmanr(dm.radio, dm[x])
    rm, pm = partial(dm.radio.values, dm[x].values, dm.logM.values)
    print(f"{'':<17} n={len(dm):2d}  rho {r:+.2f} (p={p:.3f})   "
          f"at fixed WL mass {rm:+.2f} (p={pm:.3f})")
