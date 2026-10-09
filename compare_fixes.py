"""Judge the two fixes against the current configuration (snapshots 99+91,
144 MHz, 1 Mpc radio box, X-ray L_X scale 0.278), paired by seed.

Fix 1 -- bigger radio boxes (2, 3 Mpc): mock R2 on the snapshot-99
  clusters, how much the prediction leans on mass, and the real clusters.
Fix 2 -- X-ray brightness recalibrated by the measured 1.29 (L_X scale
  0.358): mock R2, and whether the real X-ray predictions stop being
  systematically younger than the radio ones.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

SEEDS = range(42, 47)
key = lambda s: "".join(str(s).split()).replace("_", "").upper()
wl = pd.read_csv("lovoccs_wl_masses.csv").set_index("key")


def load(pat):
    out = {}
    for s in SEEDS:
        p = f"cnn_preds/{pat.format(s)}"
        if os.path.exists(p):
            out[s] = np.load(p)
    return out


def r2_99(d):
    m = d["snap"] == 99
    return r2_score(d["tsc"][m], d["oof_cluster"][m])


def rho_mass(d):
    import h5py
    m = d["snap"] == 99
    return stats.spearmanr(d["oof_cluster"][m], np.asarray(d["sim_logm"])[m])[0] \
        if "sim_logm" in d and len(d["sim_logm"]) == len(m) else np.nan


def real(ds):
    names = [key(n) for n in next(iter(ds.values()))["obs_name"]]
    P = np.mean([d["obs_pred"] for d in ds.values()], axis=0)
    return dict(zip(names, P))


def compare(label, base_pat, new_pat):
    a, b = load(base_pat), load(new_pat)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        print(f"{label}: no paired runs yet")
        return None, None
    ra = np.array([r2_99(a[s]) for s in seeds])
    rb = np.array([r2_99(b[s]) for s in seeds])
    t = stats.ttest_rel(rb, ra) if len(seeds) > 1 else None
    ma = np.nanmean([rho_mass(a[s]) for s in seeds])
    mb = np.nanmean([rho_mass(b[s]) for s in seeds])
    print(f"{label:<34} R2 {ra.mean():.3f} -> {rb.mean():.3f}  "
          f"diff {np.mean(rb - ra):+.3f} +- {np.std(rb - ra, ddof=1) if len(seeds) > 1 else 0:.3f}"
          f"{f'  p={t.pvalue:.3f}' if t is not None else ''}  (n={len(seeds)})   "
          f"rho(pred, M) {ma:+.2f} -> {mb:+.2f}")
    return {s: a[s] for s in seeds}, {s: b[s] for s in seeds}


def offset(lab, radio, xray):
    r, x = real(radio), real(xray)
    both = sorted(set(r) & set(x))
    d = np.array([r[k] - x[k] for k in both])
    rho, p = stats.spearmanr([r[k] for k in both], [x[k] for k in both])
    lm = [k for k in both if k in wl.index]
    pm = np.nan
    if len(lm) > 3:
        M = stats.rankdata([wl.loc[k, "log_m500c"] for k in lm])
        A = np.column_stack([M, np.ones(len(M))])
        res = lambda v: stats.rankdata(v) - A @ np.linalg.lstsq(A, stats.rankdata(v), rcond=None)[0]
        pm = stats.pearsonr(res([r[k] for k in lm]), res([x[k] for k in lm]))[0]
    print(f"  real, {lab:<30} radio median {np.median([r[k] for k in both]):.2f}, X-ray median "
          f"{np.median([x[k] for k in both]):.2f}, radio-X-ray {np.median(d):+.2f} Gyr "
          f"({np.mean(d > 0):.0%} radio older); rho {rho:+.2f} (p={p:.3f}), "
          f"at fixed WL mass {pm:+.2f}, n={len(both)}")


print("======== Fix 2: X-ray brightness x1.29 (L_X scale 0.278 -> 0.358)")
_, xnew = compare("X-ray, 99+91", "s9991_x_s{}_preds.npz", "lxcal_x_s{}_preds.npz")
compare("joint radio + X-ray, 99+91", "s9991_joint_s{}_preds.npz", "lxcal_joint_s{}_preds.npz")
rad = load("s9991_sfimg_s{}_preds.npz")
xold = load("s9991_x_s{}_preds.npz")
if rad and xold:
    offset("current (scale 0.278)", rad, xold)
if rad and xnew:
    offset("recalibrated (0.358)", rad, xnew)

print("\n======== Fix 1: radio box size (1 Mpc -> 2, 3 Mpc)")
for B in (2000, 3000):
    _, img = compare(f"radio, {B // 1000} Mpc box", "s9991_sfimg_s{}_preds.npz",
                     f"b{B}_sfimg_s{{}}_preds.npz")
    compare(f"radio + mass, {B // 1000} Mpc box", "s9991_sfimgmassn_s{}_preds.npz",
            f"b{B}_sfimgmassn_s{{}}_preds.npz")
    for xlab, xd in (("X-ray 0.278", xold), ("X-ray 0.358", xnew)):
        if img and xd:
            offset(f"{B // 1000} Mpc radio vs {xlab}", img, xd)
