"""Does fixing the X-ray stretch change the "X-ray doesn't help" conclusion?

Same 352 clusters, same 5 seeds, same folds, same protocol (inner-split
checkpoint selection), pseudo-TSC throughout, so every number here is
directly comparable with the radio headline (0.487 +- 0.033).

  1. X-ray CNN, original stretch vs scaled stretch (paired by seed).
  2. X-ray CNN vs four hand-computed X-ray morphology scalars -- the CNN
     should at least match them if its input is sound.
  3. Does X-ray add to radio? Stack the saved OOF predictions (and the
     morphology scalars) in a second-level ridge on the same folds, as in
     analyze_mass_control.py. Both columns are already out-of-fold, so the
     stacked number is mildly optimistic -- an upper bound on the increment.
  4. Within mass terciles, where radio's advantage has to be morphological.
"""
import glob
import re

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KF = KFold(5, shuffle=True, random_state=0)


def ridge_oof(X, y):
    X = X if X.ndim == 2 else X[:, None]
    return cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(
        alphas=np.logspace(-3, 3, 13))), X, y, cv=KF)


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        d = np.load(p, allow_pickle=True)
        out[int(re.search(r"_s(\d+)\.npz$", p).group(1))] = (d["oof_pred"],
                                                             d["halo_id"])
    return out


with h5py.File("dataset_nh4_128.h5") as f:
    hid = f["meta/halo_id"][:]
    tsc = f["labels/pseudo_tsc"][:].astype(np.float64)
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mm = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mm[h] for h in hid])

runs = {"radio": load("cnn_preds/pooled_nh4_inner_s*.npz"),
        "xray orig": load("cnn_preds/xray_orig_inner_s*.npz"),
        "xray scaled": load("cnn_preds/xray_scaled_inner_s*.npz")}
for k, v in runs.items():
    for s, (_, h) in v.items():
        assert np.array_equal(h, hid), f"{k} seed {s}: halo order differs"
seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
print(f"paired seeds: {seeds}\n")
r2 = {k: np.array([r2_score(tsc, v[s][0]) for s in seeds])
      for k, v in runs.items()}
ens = {k: np.mean([v[s][0] for s in seeds], axis=0) for k, v in runs.items()}

print("======== 1. per-seed OOF R2")
print(f"{'seed':>6}" + "".join(f"{k:>14}" for k in runs))
for i, s in enumerate(seeds):
    print(f"{s:>6}" + "".join(f"{r2[k][i]:>14.3f}" for k in runs))
print(f"{'mean':>6}" + "".join(f"{r2[k].mean():>14.3f}" for k in runs))
print(f"{'sd':>6}" + "".join(f"{r2[k].std(ddof=1):>14.3f}" for k in runs))
print(f"{'ens':>6}" + "".join(f"{r2_score(tsc, ens[k]):>14.3f}" for k in runs))
d = r2["xray scaled"] - r2["xray orig"]
print(f"\n  scaled - orig: {d.mean():+.3f} +- "
      f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE), "
      f"p={stats.ttest_rel(r2['xray scaled'], r2['xray orig']).pvalue:.4f}, "
      f"{int((d > 0).sum())}/{len(d)} positive")

# ---------- hand-computed X-ray morphology ----------
with h5py.File("dataset_xray_128_orig.h5") as f:
    lin = np.sinh(f["images"][:].astype(np.float64))
N, P, H, _ = lin.shape
yy, xx = np.mgrid[:H, :H]
rr = np.hypot(yy - H / 2 + .5, xx - H / 2 + .5)
R = H / 2
feats = np.zeros((N, 4))
for i in range(N):
    acc = []
    for p in range(P):
        im = lin[i, p]
        c = im[rr < 0.1 * R].sum() / im[rr < R].sum()
        cents = []
        for fr in np.linspace(0.1, 1.0, 10):
            m = rr < fr * R
            w = im * m
            cents.append([(w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()])
        wsh = np.array(cents).std(axis=0).sum() / R
        m = rr < 0.5 * R
        w = im * m
        cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
        ev = np.linalg.eigvalsh(np.cov(np.vstack([(yy - cy)[m], (xx - cx)[m]]),
                                       aweights=im[m] + 1e-30))
        acc.append([c, wsh, np.log10(im[rr < R].sum()),
                    1 - np.sqrt(ev[0] / ev[1])])
    feats[i] = np.mean(acc, axis=0)

print("\n======== 2. X-ray CNN vs the literature's X-ray scalars")
r2_feats = r2_score(tsc, ridge_oof(feats, tsc))
print(f"  4 morphology scalars (c, w, flux, ellipticity)  {r2_feats:.3f}")
print(f"  X-ray CNN, original stretch (ensemble)          "
      f"{r2_score(tsc, ens['xray orig']):.3f}")
print(f"  X-ray CNN, scaled stretch (ensemble)            "
      f"{r2_score(tsc, ens['xray scaled']):.3f}")

print("\n======== 3. does X-ray add to radio? (second-level ridge, same folds)")
base = r2_score(tsc, ridge_oof(ens["radio"], tsc))
rows = [("radio CNN", ens["radio"][:, None]),
        ("radio + X-ray CNN (orig)",
         np.column_stack([ens["radio"], ens["xray orig"]])),
        ("radio + X-ray CNN (scaled)",
         np.column_stack([ens["radio"], ens["xray scaled"]])),
        ("radio + X-ray scalars",
         np.column_stack([ens["radio"], feats])),
        ("radio + X-ray CNN (scaled) + scalars",
         np.column_stack([ens["radio"], ens["xray scaled"], feats])),
        ("radio + log M500", np.column_stack([ens["radio"], logm])),
        ("radio + X-ray CNN (scaled) + log M500",
         np.column_stack([ens["radio"], ens["xray scaled"], logm]))]
for name, X in rows:
    v = r2_score(tsc, ridge_oof(X, tsc))
    print(f"  {name:<40}{v:.3f}   ({v - base:+.3f} over radio)")
print(f"\n  corr(radio CNN, X-ray CNN scaled)   Spearman "
      f"{stats.spearmanr(ens['radio'], ens['xray scaled'])[0]:.3f}")
res_r = tsc - ens["radio"]
print(f"  corr(radio residual, X-ray scaled)  Spearman "
      f"{stats.spearmanr(res_r, ens['xray scaled'])[0]:+.3f}  "
      f"(does X-ray know what radio gets wrong?)")

print("\n======== 4. within mass terciles (5-seed ensembles)")
q = np.quantile(logm, [1 / 3, 2 / 3])
print(f"{'tercile':<8}{'radio':>9}{'X orig':>9}{'X scaled':>10}"
      f"{'X scalars':>11}{'rho(X scaled, M)':>18}")
print(f"{'all':<8}{'':>9}{'':>9}{'':>10}{'':>11}"
      f"{stats.spearmanr(ens['xray scaled'], logm)[0]:>18.3f}")
feat_oof = ridge_oof(feats, tsc)
for lab, b in [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
               ("high", logm > q[1])]:
    print(f"{lab:<8}{r2_score(tsc[b], ens['radio'][b]):>9.3f}"
          f"{r2_score(tsc[b], ens['xray orig'][b]):>9.3f}"
          f"{r2_score(tsc[b], ens['xray scaled'][b]):>10.3f}"
          f"{r2_score(tsc[b], feat_oof[b]):>11.3f}")
