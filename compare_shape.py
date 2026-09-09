"""B2 option 4: what is left when total flux is removed?

Pairs the shape-only runs against the full-flux runs seed for seed (same
folds, same initialisation), so the difference is attributable to the
normalisation and nothing else.

The question this answers for the paper: of the pooled CNN's skill, how
much comes from how bright the cluster is and how much from how the
emission is arranged? The full-flux model's own output is 72% explainable
by five scalars, 62% by total flux alone, so the expectation is a real drop
-- the number that matters is whether what survives is still a result.
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
    if X.ndim == 1:
        X = X[:, None]
    return cross_val_predict(
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        X, y, cv=KF)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.column_stack([b, np.ones_like(b)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return stats.pearsonr(resid(rx, rz), resid(ry, rz))[0]


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        d = np.load(p, allow_pickle=True)
        out[int(re.search(r"_s(\d+)\.npz$", p).group(1))] = d["oof_pred"]
    return out


# Scalars always come from the un-normalised dataset: "total flux" has to
# keep its physical meaning for the comparison to say anything.
with h5py.File("dataset_nh4_128.h5") as f:
    hid = f["meta/halo_id"][:]
    r500 = f["meta/r500c_kpc"][:]
    tsc = f["labels/pseudo_tsc"][:].astype(np.float64)
    imgs = f["images"][:]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
logm = np.array([mmap[h] for h in hid])
lin_tot = np.log10(np.sinh(imgs.astype(np.float64)).sum(axis=(1, 2, 3)) + 1)
SCALARS = np.column_stack([logm, np.log10(r500), lin_tot,
                           imgs.mean(axis=(1, 2, 3)),
                           (imgs > 0).mean(axis=(1, 2, 3))])

full = load("cnn_preds/pooled_nh4_inner_s*.npz")
shape = load("cnn_preds/pooled_shape_inner_s*.npz")
seeds = sorted(set(full) & set(shape))
print(f"paired seeds: {seeds}\n")

print("======== per seed, OOF R2")
print(f"{'seed':>6}{'full flux':>12}{'shape only':>12}{'delta':>10}")
for s in seeds:
    a, b = r2_score(tsc, full[s]), r2_score(tsc, shape[s])
    print(f"{s:>6}{a:>12.3f}{b:>12.3f}{b - a:>10.3f}")
fa = np.array([r2_score(tsc, full[s]) for s in seeds])
fb = np.array([r2_score(tsc, shape[s]) for s in seeds])
d = fb - fa
t, p = stats.ttest_rel(fb, fa)
print(f"\n  full  {fa.mean():.3f} +- {fa.std(ddof=1):.3f}")
print(f"  shape {fb.mean():.3f} +- {fb.std(ddof=1):.3f}")
print(f"  paired shape - full: {d.mean():+.3f} +- "
      f"{d.std(ddof=1) / np.sqrt(len(d)):.3f} (SE), p={p:.4f}, "
      f"{int((d < 0).sum())}/{len(d)} negative")

f_ens = np.mean([full[s] for s in seeds], axis=0)
s_ens = np.mean([shape[s] for s in seeds], axis=0)
print(f"\n  5-seed ensemble: full {r2_score(tsc, f_ens):.3f}   "
      f"shape {r2_score(tsc, s_ens):.3f}")

print("\n======== what is the shape-only model keyed on?")
r2_scalars = r2_score(tsc, ridge_oof(SCALARS, tsc))
for name, pred in [("full flux", f_ens), ("shape only", s_ens)]:
    print(f"  {name:<12} rho(pred, TSC) {stats.spearmanr(pred, tsc)[0]:+.3f}"
          f"   rho(pred, M500) {stats.spearmanr(pred, logm)[0]:+.3f}"
          f"   rho(pred, total flux) {stats.spearmanr(pred, lin_tot)[0]:+.3f}"
          f"   partial(TSC|M) {partial_spearman(pred, tsc, logm):+.3f}")
    print(f"  {'':<12} explained by the 5 scalars: R2 "
          f"{r2_score(pred, ridge_oof(SCALARS, pred)):+.3f}"
          f"   by total flux alone: "
          f"{r2_score(pred, ridge_oof(lin_tot, pred)):+.3f}")
    inc = r2_score(tsc, ridge_oof(np.column_stack([SCALARS, pred]), tsc))
    print(f"  {'':<12} scalars + this model: {inc:+.3f} "
          f"(increment {inc - r2_scalars:+.3f} over scalars alone "
          f"at {r2_scalars:.3f})\n")

print("======== R2 within mass terciles")
q = np.quantile(logm, [1 / 3, 2 / 3])
bins = [("low", logm <= q[0]), ("mid", (logm > q[0]) & (logm <= q[1])),
        ("high", logm > q[1])]
print(f"{'bin':<7}{'n':>5}{'full R2':>10}{'shape R2':>10}"
      f"{'full rho':>10}{'shape rho':>11}")
for name, b in bins:
    print(f"{name:<7}{b.sum():>5}{r2_score(tsc[b], f_ens[b]):>10.3f}"
          f"{r2_score(tsc[b], s_ens[b]):>10.3f}"
          f"{stats.spearmanr(f_ens[b], tsc[b])[0]:>10.3f}"
          f"{stats.spearmanr(s_ens[b], tsc[b])[0]:>11.3f}")
