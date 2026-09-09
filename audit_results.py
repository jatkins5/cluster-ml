"""Pre-paper audit: things that would be embarrassing to discover in review.

1. Label agreement: pseudo-TSC (the headline CNN target) vs the merger-catalog
   tsc_gyr (the physically defined one). If they disagree the paper needs to
   pick one and justify it.
2. Trivial confounds: how much of pseudo-TSC is predictable from halo mass or
   from total image brightness alone? A CNN R2 that a one-number baseline
   nearly matches is not a morphology result.
3. Shrinkage: OOF predictions on held-out mocks have some spread; if the real
   LoTSS predictions have the same spread the "compression" worry is just
   ordinary regression-to-the-mean at R2~0.4, not a transfer failure.
4. Cross-run agreement on the 17 real clusters: independently trained
   configurations (seeds, noise models, cuts) should rank the real clusters
   the same way if there is signal. Spearman between every pair of runs.
"""
import glob
import re

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score


def sec(t):
    print(f"\n{'=' * 8} {t}")


# ---------- 1. labels ----------
sec("label agreement")
with h5py.File("dataset_nh4_128.h5") as f:
    hid = f["meta/halo_id"][:]
    r500 = f["meta/r500c_kpc"][:]
    ptsc = f["labels/pseudo_tsc"][:]
    imgs = f["images"][:]                       # (N,3,H,W) arcsinh(w)
with h5py.File("TSC_Cutimages/TSC_eachhalo_snap99.hdf5") as f:
    tmap = dict(zip(f["halo_id"][:], f["tsc_gyr"][:]))
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    mmap = dict(zip(f["haloID"][:], f["mhalo_500c"][:]))
mtsc = np.array([tmap.get(h, np.nan) for h in hid])
logm = np.array([mmap[h] for h in hid])
ok = np.isfinite(mtsc)
capped = ptsc >= 3.999
print(f"N={len(hid)}  pseudo-TSC capped at 4.0: {capped.sum()}  "
      f"merger-TSC NaN: {(~ok).sum()}")
print(f"pseudo-TSC: median {np.median(ptsc):.2f}  sd {ptsc.std():.2f}")
print(f"merger-TSC: median {np.nanmedian(mtsc):.2f}  sd {np.nanstd(mtsc):.2f}"
      f"  max {np.nanmax(mtsc):.2f}")
rho, p = stats.spearmanr(ptsc[ok], mtsc[ok])
print(f"Spearman(pseudo, merger) = {rho:.3f} (p={p:.1e})   "
      f"Pearson = {stats.pearsonr(ptsc[ok], mtsc[ok])[0]:.3f}")
u = ok & ~capped
print(f"  uncensored only (n={u.sum()}): Spearman = "
      f"{stats.spearmanr(ptsc[u], mtsc[u])[0]:.3f}")
# does capping hide old mergers?
print(f"  merger-TSC of capped clusters: median {np.nanmedian(mtsc[capped]):.2f}"
      f" vs uncapped {np.nanmedian(mtsc[~capped & ok]):.2f}")

# ---------- 2. confounds ----------
sec("one-number baselines for pseudo-TSC (5-fold OOF ridge)")
lin_tot = np.log10(np.sinh(imgs.astype(np.float64)).sum(axis=(1, 2, 3)) + 1)
arc_mean = imgs.mean(axis=(1, 2, 3))
frac_pos = (imgs > 0).mean(axis=(1, 2, 3))
feats = {
    "log M500": logm,
    "log r500": np.log10(r500),
    "log total linear w": lin_tot,
    "mean arcsinh pixel": arc_mean,
    "filled fraction": frac_pos,
}
kf = KFold(5, shuffle=True, random_state=0)
for name, x in feats.items():
    rho = stats.spearmanr(x, ptsc)[0]
    pred = cross_val_predict(RidgeCV(), x[:, None], ptsc, cv=kf)
    print(f"{name:<22} Spearman {rho:+.3f}   OOF R2 {r2_score(ptsc, pred):+.3f}")
X = np.column_stack(list(feats.values()))
pred = cross_val_predict(RidgeCV(), X, ptsc, cv=kf)
print(f"{'all five together':<22} {'':>15}   OOF R2 {r2_score(ptsc, pred):+.3f}")
if ok.sum():
    pred = cross_val_predict(RidgeCV(), X[ok], mtsc[ok], cv=kf)
    print(f"  (same five -> merger-TSC: OOF R2 {r2_score(mtsc[ok], pred):+.3f})")

# ---------- 3 + 4. transfer runs ----------
sec("shrinkage: OOF mock prediction spread vs real prediction spread")
runs = {}
for p in sorted(glob.glob("*_preds.npz")):
    d = np.load(p, allow_pickle=True)
    if "obs_pred" not in d:
        continue
    runs[p[:-10]] = d
    oc, t = d["oof_cluster"], d["tsc"]
    slope = np.polyfit(t, oc, 1)[0]
    print(f"{p[:-10]:<24} mock R2 {r2_score(t, oc):+.3f}  "
          f"label sd {t.std():.2f}  OOF pred sd {oc.std():.2f}  "
          f"slope(pred|label) {slope:.2f}  |  real pred sd {d['obs_pred'].std():.2f}"
          f"  median {np.median(d['obs_pred']):.2f}")

sec("cross-run Spearman of the 17 real-cluster predictions")
names = list(runs)
ref_names = runs[names[0]]["obs_name"]
for n in names:
    assert list(runs[n]["obs_name"]) == list(ref_names), n
M = np.eye(len(names))
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        M[i, j] = M[j, i] = stats.spearmanr(runs[names[i]]["obs_pred"],
                                            runs[names[j]]["obs_pred"])[0]
w = max(len(n) for n in names)
print(" " * w + "".join(f"{k:>7}" for k in range(len(names))))
for i, n in enumerate(names):
    print(f"{n:<{w}}" + "".join(f"{M[i, j]:>7.2f}" for j in range(len(names))))
off = M[np.triu_indices(len(names), 1)]
print(f"mean off-diagonal Spearman: {off.mean():.2f}  min {off.min():.2f}")
# per-cluster consensus rank
ranks = np.mean([stats.rankdata(runs[n]["obs_pred"]) for n in names], axis=0)
order = np.argsort(ranks)
print("\nconsensus ordering (youngest first): " +
      ", ".join(f"{ref_names[i]}({ranks[i]:.1f})" for i in order))

# ---------- 5. how big is the best-epoch selection effect ----------
sec("best-epoch vs final-epoch fold R2 from logs")
pat_final = re.compile(r"Epoch\s+60/60.*val R²=([-\d.]+)")
pat_best = re.compile(r"best → R²=([-\d.]+)")
for tag, g in [("pooled/shallow baselines (recenter_seed)",
                "logs/recenter_seed_*.out"),
               ("density-cut CNN grid (cut_cnn)", "logs/cut_cnn_*.out"),
               ("injected transfer", "logs/inject_transfer_*.out"),
               ("regen transfer", "logs/regen_transfer_*.out")]:
    fin, best = [], []
    for fn in glob.glob(g):
        txt = open(fn).read()
        f_ = [float(x) for x in pat_final.findall(txt)]
        b_ = [float(x) for x in pat_best.findall(txt)]
        if not f_:   # mock scripts print a different line
            f_ = [float(x) for x in re.findall(
                r"epoch\s+60/60\s+val R2=([-+\d.]+)", txt)]
            b_ = [float(x) for x in re.findall(
                r"fold \d best val R2 = ([-+\d.]+)", txt)]
        if len(f_) == len(b_) and f_:
            fin += f_
            best += b_
    if fin:
        fin, best = np.array(fin), np.array(best)
        d = best - fin
        print(f"{tag:<42} n_folds={len(d):3d}  best-final: mean {d.mean():+.3f}"
              f"  median {np.median(d):+.3f}  max {d.max():+.3f}")
